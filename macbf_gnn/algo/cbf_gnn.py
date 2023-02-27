import torch.nn as nn
import os
import torch
import random
import numpy as np
import cvxpy as cp

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Sequential
from torch import Tensor
from torch.optim import Adam
from torch.autograd.functional import jacobian
from torch_geometric.nn.conv.transformer_conv import TransformerConv
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
        # self.feat_transformer = Sequential('x, edge_index, edge_attr', [
        #     (TransformerConv(in_channels=node_dim, out_channels=128, edge_dim=edge_dim),
        #      'x, edge_index, edge_attr -> x'),
        #     nn.ReLU(),
        #     (TransformerConv(in_channels=128, out_channels=64, edge_dim=edge_dim),
        #      'x, edge_index, edge_attr -> x'),
        #     # nn.ReLU(),
        #     # (TransformerConv(in_channels=128, out_channels=64, edge_dim=edge_dim),
        #     #  'x, edge_index, edge_attr -> x'),
        # ])
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
            self.params = {  #TODO: tune this
                'alpha': 0.5,
                'eps': 0.02,
                'inner_iter': 10,
                'loss_action_coef': 0.1,
                'loss_unsafe_coef': 10.,
                'loss_safe_coef': 10.,
                'loss_h_dot_coef': 50.
            }
        else:
            self.params = params

    def act(self, data: Data) -> Tensor:
        with torch.no_grad():
            return self.actor(data)
        # return self.act_strict(data)

    def step(self, data: Data) -> Tensor:
        action = self.actor(data)
        is_safe = True
        if torch.any(self._env.unsafe_mask(data)):
            is_safe = not is_safe
            
        self.buffer.append(data, is_safe)
        
        return action

    def is_update(self, step: int) -> bool:
        return step % self.batch_size == 0

    def update(self, step: int, writer: SummaryWriter = None) -> dict:
        seg_len = 3  # pls use odd number
        acc_safe = torch.zeros(1, dtype=torch.float)
        acc_unsafe = torch.zeros(1, dtype=torch.float)
        acc_h_dot = torch.zeros(1, dtype=torch.float)
        for i_inner in range(self.params['inner_iter']):
            # sample from the current buffer and the memory
            if self.memory.size == 0:
                graphs = Batch.from_data_list(self.buffer.sample(self.batch_size // 5, seg_len))
                
            else:
                curr_graphs = self.buffer.sample(self.batch_size // 10, seg_len, True)
                prev_graphs = self.memory.sample(self.batch_size // 5 - self.batch_size // 10, seg_len, True)
                graphs = Batch.from_data_list(curr_graphs + prev_graphs)

            # get CBF values and the control inputs
            h = self.cbf(graphs)
            actions = self.actor(graphs)

            # calculate loss
            eps = self.params['eps']
            # unsafe region h(x) < 0
            unsafe_mask = self._env.unsafe_mask(graphs)
            h_unsafe = h[unsafe_mask]
            if h_unsafe.numel():
                max_val_unsafe = torch.relu(h_unsafe + eps)
                loss_unsafe = torch.mean(max_val_unsafe)  # use square loss for robustness
                acc_unsafe = torch.mean(torch.less(h_unsafe, 0).type_as(h_unsafe))
                
            else:
                loss_unsafe = torch.tensor(0.0).type_as(h_unsafe)
                acc_unsafe = torch.tensor(1.0).type_as(h_unsafe)
                
            # safe region h(x) > 0
            safe_mask = self._env.safe_mask(graphs)
            h_safe = h[safe_mask]
            if h_safe.numel():
                max_val_safe = torch.relu(-h_safe + eps)
                loss_safe = torch.mean(max_val_safe)  # use square loss for robustness
                acc_safe = torch.mean(torch.greater_equal(h_safe, 0).type_as(h_safe))
                
            else:
                loss_safe = torch.tensor(0.0).type_as(h_unsafe)
                acc_safe = torch.tensor(1.0).type_as(h_unsafe)
                
            # derivative loss h_dot + \alpha h > 0
            graphs_next = self._env.forward_graph(graphs, actions)  # todo: change edge attr
            h_next = self.cbf(graphs_next)
            h_dot = (h_next - h) / self._env.dt
            max_val_h_dot = torch.relu(-h_dot - self.params['alpha'] * h + eps)
            # loss_h_dot = torch.sum(max_val_h_dot) / torch.sum(torch.greater_equal(h_dot, 0).type_as(h_dot))
            loss_h_dot = torch.mean(max_val_h_dot)  # use square loss for robustness
            acc_h_dot = torch.mean(torch.greater_equal(h_dot + self.params['alpha'] * h, 0).type_as(h_dot))
            # action loss
            loss_action = torch.mean(torch.square(actions).sum(dim=1))

            # backpropagation
            loss = self.params['loss_unsafe_coef'] * loss_unsafe + \
                   self.params['loss_safe_coef'] * loss_safe + \
                   self.params['loss_h_dot_coef'] * loss_h_dot + \
                   self.params['loss_action_coef'] * loss_action
            self.optim_cbf.zero_grad()
            self.optim_actor.zero_grad()
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
        # set up variable and constraints
        action = cp.Variable((data.states.shape[0], self.action_dim))
        relaxation = cp.Variable(1, nonneg=True)
        obj = cp.Minimize(cp.sum_squares(action) + cp.multiply(1e4, relaxation))
        constraints = []

        # calculate Jacobian of h
        data.edge_attr.requires_grad = True
        h = self.cbf(data)
        Jh = []
        for i in range(h.shape[0]):
            Jh.append(torch.autograd.grad(h[i], data.edge_attr, create_graph=True, retain_graph=True)[0].unsqueeze(0))
        Jh = torch.cat(Jh, dim=0).cpu().detach().numpy().reshape(h.shape[0], -1)
        h = h.cpu().detach().numpy()

        # calculate h_dot
        # for i in range(self.num_agents):
        #     action =
        edge_dot = self._env.edge_dynamics(data, action)
        edge_dot = cp.reshape(edge_dot, (1, edge_dot.shape[0] * edge_dot.shape[1]), order='C')
        for i in range(h.shape[0]):
            constraints.append(cp.scalar_product(Jh[i], edge_dot) + self.params['alpha'] * h[i] >= -relaxation)

        # action limit
        low, high = self._env.action_lim
        constraints.append(action >= np.repeat(low.unsqueeze(0).cpu().numpy(), self.num_agents, axis=0))
        constraints.append(action <= np.repeat(high.unsqueeze(0).cpu().numpy(), self.num_agents, axis=0))

        # solve problem
        # constraints = [h_dot + self.params['alpha'] * h.squeeze().cpu().detach().numpy() >= 0]
        prob = cp.Problem(obj, constraints)
        results = prob.solve(solver='SCS')
        action = action.value
        # if relaxation.value > 0:
        #     print(f'relaxation: {relaxation.value}')

        return torch.from_numpy(action).type_as(data.x)


        # with torch.no_grad():
        #     h = self.cbf(data)
        # # with torch.no_grad():
        # #     action = self.actor(data)
        # # action = torch.zeros_like(action, requires_grad=True)
        # # learned_action = False
        # i_iter = 0
        # # optim = None
        # optim = []
        # action = []
        # for i in range(self.num_agents):
        #     action.append(torch.zeros(1, self.action_dim, dtype=torch.float, device=self.device, requires_grad=True))
        #     optim.append(Adam((action[-1],), lr=0.1))
        # # action_tensor = torch.cat(action, dim=0)
        # # optim = Adam((action,), lr=0.1)
        # while True:
        #     action_tensor = torch.cat(action, dim=0)
        #     action_ref = self._env.u_ref(data)
        #     graphs_next = self._env.forward_graph(data, action_tensor)
        #     h_next = self.cbf(graphs_next)
        #     h_dot = (h_next - h) / self._env.dt
        #     val_h_dot = torch.relu(-h_dot - self.params['alpha'] * h)
        #     val_agent = torch.nonzero(val_h_dot.squeeze(-1))
        #     max_val_h_dot = torch.mean(val_h_dot)
        #     if max_val_h_dot <= 0 or i_iter >= 10000:
        #         break
        #     else:
        #         # if learned_action is False:
        #         #     with torch.no_grad():
        #         #         action = self.actor(data)
        #         #     action.requires_grad = True
        #         #     optim = Adam((action,), lr=0.1)
        #         #     learned_action = True
        #         # else:
        #         #     optim.zero_grad()
        #         #     max_val_h_dot.backward()
        #         #     optim.step()
        #         #     i_iter += 1
        #         for i in val_agent[0]:
        #             optim[i].zero_grad()
        #         max_val_h_dot.backward()
        #         for i in val_agent[0]:
        #             # action[i].grad += 10 * torch.randn_like(action[i]) * action[i].grad
        #             true_action = action[i] + action_ref[i, :]
        #             if torch.dot(true_action.squeeze(), action[i].grad.squeeze()) / (torch.norm(true_action) * torch.norm(action[i].grad) + 1e-8) > 0.9:
        #                 if action[i].grad.ndim == 2:
        #                     action[i].grad = torch.tensor([action[i].grad[0, 1], -action[i].grad[0, 0]]).type_as(action[i]).view(action[i].grad.shape)
        #                 else:
        #                     rand_vec = torch.randn_like(action[i])
        #                     if torch.dot(rand_vec, action[i].grad) == 0:
        #                         rand_vec = torch.randn_like(action[i])
        #                     new_grad = torch.cross(action[i].grad, rand_vec)
        #                     new_grad = new_grad / torch.norm(new_grad) * torch.norm(action[i].grad)
        #                     action[i].grad = new_grad
        #             optim[i].step()
        #         i_iter += 1
        #
        # # print(i_iter)
        # return torch.cat(action, dim=0)

        # h = self.cbf(data)
        # actions = []
        # for i_node in range(self.num_agents):
        #     action = cp.Variable(self.action_dim)
        #     obj = cp.Minimize(cp.sum_squares(action))
        #     graph_next = self._env.forward_agent(data, action, i_node)
        #     h_next = self.cbf(graph_next)
        #     h_dot = (h_next - h) / self._env.dt
        #     max_val_h_dot = torch.mean(torch.relu(-h_dot - self.params['alpha'] * h))
        #     constraints = [max_val_h_dot <= 0]
        #     prob = cp.Problem(obj, constraints)
        #     result = prob.solve()
        #     actions.append(torch.tensor(action.value).type_as(data.states))
        # return torch.cat(actions)
        # actions = cp.Variable(self.action_dim * self.num_agents)
        # obj =
