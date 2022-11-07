import torch.nn as nn
import os
import torch

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Sequential
from torch import Tensor
from torch.optim import Adam

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
            (CBFGNNLayer(node_dim=node_dim, edge_dim=edge_dim, output_dim=64, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
            nn.ReLU(),
            (CBFGNNLayer(node_dim=64, edge_dim=edge_dim, output_dim=64, phi_dim=phi_dim),
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
        h: (bs, n)
            CBF values for all agents
        """
        x = self.feat_transformer(data.x, data.edge_attr, data.edge_index)
        h = self.feat_2_CBF(x)
        return h
        # return h.reshape(-1, self.num_agents)


class MACBFGNN(Algorithm):

    def __init__(
            self,
            env: MultiAgentEnv,
            num_agents: int,
            node_dim: int,
            edge_dim: int,
            action_dim: int,
            device: torch.device,
            batch_size: int = 500
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
        self.optim_actor = Adam(self.actor.parameters(), lr=3e-4)

        # buffer to store data used in training
        self.buffer = Buffer()  # buffer for current episode
        self.memory = Buffer()  # replay buffer
        self.batch_size = batch_size

        # hyperparams
        self.params = {
            'alpha': 1.0,
            'eps': 0.01,
            'inner_iter': 10,
            'loss_action_coef': 0.1
        }

    def act(self, data: Data) -> Tensor:
        with torch.no_grad():
            return self.actor(data)

    def step(self, data: Data) -> Tensor:
        action = self.actor(data)
        self.buffer.append(data)
        return action

    def is_update(self, step: int) -> bool:
        return step % self.batch_size == 0

    def update(self, step: int, writer: SummaryWriter = None):
        for i_inner in range(self.params['inner_iter']):
            # sample from the current buffer and the memory
            if self.memory.size == 0:
                graphs = Batch.from_data_list(self.buffer.sample(self.batch_size // 5))
            else:
                curr_graphs = self.buffer.sample(self.batch_size // 10)
                prev_graphs = self.memory.sample(self.batch_size // 5 - self.batch_size // 10)
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
                max_val_unsafe = torch.maximum(h_unsafe + eps, torch.zeros_like(h_unsafe))
                loss_unsafe = torch.mean(max_val_unsafe**2)  # use square loss for robustness
                acc_unsafe = torch.mean(torch.less(h_unsafe, 0).type_as(h_unsafe))
            else:
                loss_unsafe = torch.tensor(0.0).type_as(h_unsafe)
                acc_unsafe = torch.tensor(1.0).type_as(h_unsafe)
            # safe region h(x) > 0
            safe_mask = self._env.safe_mask(graphs)
            h_safe = h[safe_mask]
            if h_safe.numel():
                max_val_safe = torch.maximum(-h_safe + eps, torch.zeros_like(h_safe))
                loss_safe = torch.mean(max_val_safe**2)  # use square loss for robustness
                acc_safe = torch.mean(torch.greater_equal(h_safe, 0).type_as(h_safe))
            else:
                loss_safe = torch.tensor(0.0).type_as(h_unsafe)
                acc_safe = torch.tensor(1.0).type_as(h_unsafe)
            # derivative loss h_dot + \alpha h > 0
            graphs_next = self._env.forward_graph(graphs, actions)  # todo: change edge attr
            h_next = self.cbf(graphs_next)
            h_dot = (h_next - h) / self._env.dt
            max_val_h_dot = torch.maximum(-h_dot - self.params['alpha'] * h + eps, torch.zeros_like(h_dot))
            loss_h_dot = torch.mean(max_val_h_dot**2)  # use square loss for robustness
            acc_h_dot = torch.mean(torch.greater_equal(h_dot, 0).type_as(h_dot))
            # action loss
            loss_action = torch.mean(torch.square(actions).sum(dim=1))

            # backpropagation
            loss = loss_unsafe + loss_safe + loss_h_dot + self.params['loss_action_coef'] * loss_action
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

    def save(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(self.cbf.state_dict(), os.path.join(save_dir, 'cbf.pkl'))
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pkl'))

    def load(self, load_dir: str):
        assert os.path.exists(load_dir)
        self.cbf.load_state_dict(torch.load(os.path.join(load_dir, 'cbf.pkl'), map_location=self.device))
