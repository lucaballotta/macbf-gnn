import torch.nn as nn
import os
import torch
import numpy as np
import cvxpy as cp

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Sequential
from torch import Tensor
from torch.optim import Adam
from typing import Optional

from macbf_gnn.nn import MLP, CBFGNNLayer
from macbf_gnn.controller import GNNController
from macbf_gnn.env import MultiAgentEnv

from .base import Algorithm
from .buffer import Buffer


class CBFGNN(nn.Module):

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, phi_dim: int):
        super(CBFGNN, self).__init__()
        self.num_agents = num_agents
        self.feat_transformer = Sequential('x, edge_attr, edge_index', [
            (CBFGNNLayer(node_dim=node_dim, edge_dim=edge_dim, output_dim=512, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
            nn.ReLU(),
            (CBFGNNLayer(node_dim=512, edge_dim=edge_dim, output_dim=128, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
            nn.ReLU(),
            (CBFGNNLayer(node_dim=128, edge_dim=edge_dim, output_dim=64, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
        ])
        self.feat_2_CBF = MLP(in_channels=64, out_channels=1, hidden_layers=(64, 64))


    def forward(self, data: Data) -> Tensor:
        """
        Get the CBF value for the input states.

        Parameters
        ----------
        data: Data
            batched data using Batch.from_data_list().

        Returns
        -------
        h: (bs x n,)
            CBF values for all agents
        """
        x = self.feat_transformer(data.x, data.edge_attr, data.edge_index)
        h = self.feat_2_CBF(x)
        return h

    def forward_explict(self, x: Tensor, edge_index: Tensor) -> Tensor:
        pass


class MACBFGNN(Algorithm):

    def __init__(
            self,
            env: MultiAgentEnv,
            num_agents: int,
            node_dim: int,
            edge_dim: int,
            action_dim: int,
            device: torch.device,
            batch_size: int = 500,
            params: Optional[dict] = None
    ):
        super(MACBFGNN, self).__init__(
            env=env,
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            device=device
        )

        # models
        self.cbf = CBFGNN(
            num_agents=self.num_agents,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            phi_dim=32
        ).to(device)
        self.actor = GNNController(
            num_agents=self.num_agents,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            phi_dim=32,
            action_dim=self.action_dim
        ).to(device)

        # optimizer
        self.optim_cbf = Adam(self.cbf.parameters(), lr=3e-4)
        self.optim_actor = Adam(self.actor.parameters(), lr=1e-3)

        # buffer to store data used in training
        self.buffer = Buffer()  # buffer for current episode
        self.memory = Buffer()  # replay buffer
        self.batch_size = batch_size

        # hyperparams
        if params is None:
            self.params = {  # default hyper-parameters
                'alpha': 1.0,
                'eps': 0.02,
                'inner_iter': 10,
                'loss_action_coef': 0.001,
                'loss_unsafe_coef': 1.,
                'loss_safe_coef': 1.,
                'loss_h_dot_coef': 0.1
            }
        else:
            self.params = params


    @torch.no_grad()
    def act(self, data: Data) -> Tensor:
        # with torch.no_grad():
        #     return self.actor(data)
        return self.actor(data)


    @torch.no_grad()
    def step(self, data: Data) -> Tensor:
        if data.edge_attr.numel():
            action = self.actor(data)
        else:
            action = 0
            
        is_safe = True
        if torch.any(self._env.unsafe_mask(data)):
            is_safe = not is_safe
            
        self.buffer.append(data, is_safe)
        
        return action


    def is_update(self, step: int) -> bool:
        return step % self.batch_size == 0


    def update(self, step: int, writer: SummaryWriter = None) -> dict:
        acc_safe = torch.zeros(1, dtype=torch.float)
        acc_unsafe = torch.zeros(1, dtype=torch.float)
        acc_h_dot = torch.zeros(1, dtype=torch.float)
        for i_inner in range(self.params['inner_iter']):
            
            # sample from the current buffer and the memory
            if self.memory.size == 0:
                graph_list = self.buffer.sample(self.batch_size // 5)
                
            else:
                curr_graphs = self.buffer.sample(self.batch_size // 10, True)
                prev_graphs = self.memory.sample(self.batch_size // 5 - self.batch_size // 10, True)
                graph_list = curr_graphs + prev_graphs
                
            # get CBF values and the control inputs
            graphs = Batch.from_data_list(graph_list)
            h = self.cbf(graphs)
            actions = self.actor(graphs)

            # calculate loss
            eps = self.params['eps']

            # unsafe region h(x) < 0
            unsafe_mask = self._env.unsafe_mask(graphs)
            h_unsafe = h[unsafe_mask]
            if h_unsafe.numel():
                max_val_unsafe = torch.relu(h_unsafe + eps)
                loss_unsafe = torch.mean(max_val_unsafe)
                acc_unsafe = torch.mean(torch.less(h_unsafe, 0).type_as(h_unsafe))
                
            else:
                loss_unsafe = torch.tensor(0.0).type_as(h_unsafe)
                acc_unsafe = torch.tensor(1.0).type_as(h_unsafe)
                
            # safe region h(x) > 0
            safe_mask = self._env.safe_mask(graphs)
            h_safe = h[safe_mask]
            if h_safe.numel():
                max_val_safe = torch.relu(-h_safe + eps)
                loss_safe = torch.mean(max_val_safe)
                acc_safe = torch.mean(torch.greater_equal(h_safe, 0).type_as(h_safe))
                
            else:
                loss_safe = torch.tensor(0.0).type_as(h_unsafe)
                acc_safe = torch.tensor(1.0).type_as(h_unsafe)
                
            # derivative loss h_dot + \alpha h > 0
            graphs_next = self._env.forward_graph(graphs, actions)
            h_next = self.cbf(graphs_next)
            graphs_next = []
            for i_graph, this_graph in enumerate(graph_list):
                this_graph_next = self._env.forward_graph(this_graph, actions[i_graph * self.num_agents: (i_graph + 1) * self.num_agents])
                graphs_next.append(this_graph_next)
                
            graphs_next = Batch.from_data_list(graphs_next)
            h_next_new_link = self.cbf(graphs_next)
            h_dot = (h_next - h) / self._env.dt
            h_dot_new_link = (h_next_new_link - h) / self._env.dt
            residue = (h_dot_new_link - h_dot).clone().detach()
            h_dot = residue + h_dot
            max_val_h_dot = torch.relu(-h_dot - self.params['alpha'] * h + eps)
            loss_h_dot = torch.mean(max_val_h_dot)
            acc_h_dot = torch.mean(torch.greater_equal(h_dot + self.params['alpha'] * h, 0).type_as(h_dot))
            
            # action loss
            loss_action = torch.mean(torch.square(actions).sum(dim=1))

            # backpropagation
            loss = self.params['loss_unsafe_coef'] * loss_unsafe + \
                   self.params['loss_safe_coef'] * loss_safe + \
                   self.params['loss_h_dot_coef'] * loss_h_dot + \
                   self.params['loss_action_coef'] * loss_action
            self.optim_cbf.zero_grad(set_to_none=True)
            self.optim_actor.zero_grad(set_to_none=True)
            loss.backward()
            self.optim_cbf.step()
            self.optim_actor.step()

            # save loss
            writer.add_scalar('loss/unsafe', loss_unsafe.item(), step * self.params['inner_iter'] + i_inner)
            writer.add_scalar('loss/safe', loss_safe.item(), step * self.params['inner_iter'] + i_inner)
            writer.add_scalar('loss/derivative', loss_h_dot.item(), step * self.params['inner_iter'] + i_inner)
            writer.add_scalar('loss/action', loss_action.item(), step * self.params['inner_iter'] + i_inner)

            # save accuracy
            writer.add_scalar('acc/unsafe', acc_unsafe.item(), step * self.params['inner_iter'] + i_inner)
            writer.add_scalar('acc/safe', acc_safe.item(), step * self.params['inner_iter'] + i_inner)
            writer.add_scalar('acc/derivative', acc_h_dot.item(), step * self.params['inner_iter'] + i_inner)

        # merge the current buffer to the memory
        self.memory.merge(self.buffer)
        self.buffer.clear()

        return {
            'acc/safe': acc_safe.item(),
            'acc/unsafe': acc_unsafe.item(),
            'acc/derivative': acc_h_dot.item()
        }
        
        
    def save(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(self.cbf.state_dict(), os.path.join(save_dir, 'cbf.pkl'))
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pkl'))


    def load(self, load_dir: str):
        assert os.path.exists(load_dir)
        self.cbf.load_state_dict(torch.load(os.path.join(load_dir, 'cbf.pkl'), map_location=self.device))
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, 'actor.pkl'), map_location=self.device))


    def apply(self, data: Data) -> Tensor:
        margins = np.zeros(data.states.shape[0])

        # calculate the Jacobian of h
        data.edge_attr.requires_grad = True
        h = self.cbf(data)
        Jh = []
        for i in range(h.shape[0]):
            Jh.append(torch.autograd.grad(h[i], data.edge_attr, create_graph=True, retain_graph=True)[0].unsqueeze(0))
        Jh = torch.cat(Jh, dim=0).cpu().detach().numpy().reshape(h.shape[0], -1)
        h_np = h.cpu().detach().numpy()

        i_iter = 0
        while True:
            # set up variable and constraints
            action = cp.Variable((data.states.shape[0], self.action_dim))
            relaxation = cp.Variable(1, nonneg=True)
            obj = cp.Minimize(cp.sum_squares(action) + cp.multiply(1e4, relaxation))
            constraints = []

            # calculate h_dot using linearization
            edge_dot = self._env.edge_dynamics(data, action)
            edge_dot = cp.reshape(edge_dot, (1, edge_dot.shape[0] * edge_dot.shape[1]), order='C')
            for i in range(h.shape[0]):
                constraints.append(
                    cp.scalar_product(Jh[i], edge_dot) + self.params['alpha'] * h_np[i] + relaxation >= margins[i])

            # action limit
            low, high = self._env.action_lim
            constraints.append(action >= np.repeat(low.unsqueeze(0).cpu().numpy(), self.num_agents, axis=0))
            constraints.append(action <= np.repeat(high.unsqueeze(0).cpu().numpy(), self.num_agents, axis=0))

            # solve problem
            prob = cp.Problem(obj, constraints)
            results = prob.solve(solver='SCS')
            action = torch.from_numpy(action.value).type_as(data.x)

            # simulate the environment to check CBF conditions
            graphs_next = self._env.forward_graph(data, action)
            graphs_next = self._env.add_communication_links(graphs_next)
            h_next = self.cbf(graphs_next)
            h_dot = (h_next - h) / self._env.dt
            val_h_dot = torch.relu(-h_dot - self.params['alpha'] * h)
            val_agent = torch.nonzero(val_h_dot.squeeze(-1))
            max_val_h_dot = torch.mean(val_h_dot)

            if max_val_h_dot <= 0 or i_iter >= 500:
                break
            else:
                for i in val_agent.squeeze(-1):
                    margins[i] += 0.02  # increase margin and solve again
                i_iter += 1

        return action