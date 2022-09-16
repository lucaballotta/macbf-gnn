import torch
import torch_geometric.nn as gnn
from torch_scatter import scatter, scatter_add, scatter_softmax
from torch.nn import LSTM
from utils import *
from config import *
from torch import Tensor
from torch_geometric.data import Data


# class GNN(nn.Module):
#
#     def __init__(self, feat_dim: int, phi_dim: int, aggr_dim: int, num_agents: int, state_dim: int):
#         super().__init__()
#         self.feat_transformer = gnn.Sequential('x, states', [
#             (GNNLayer(feat_dim, phi_dim, aggr_dim, num_agents, state_dim), 'x, states -> x'),
#             nn.ReLU(),
#             (GNNLayer(feat_dim, phi_dim, aggr_dim, num_agents, state_dim), 'x, states -> x'),
#             ])
#         self.feat_2_CBF = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(feat_dim, 1)
#             )
#         self.x_init = torch.ones((num_agents, feat_dim))
#
#     def forward(self, states):
#         x = self.feat_transformer(self.x_init, states)
#         h = self.feat_2_CBF(x)
#         return h


class CBFGNN(nn.Module):

    def __init__(self, state_dim: int, phi_dim: int, num_agents: int):
        super().__init__()
        self.num_agents = num_agents
        self.feat_transformer = gnn.Sequential('x, edge_attr, edge_index', [
            (CBFGNNLayer(node_dim=state_dim, edge_dim=state_dim, output_dim=64, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
            nn.ReLU(),
            (CBFGNNLayer(node_dim=64, edge_dim=state_dim, output_dim=64, phi_dim=phi_dim),
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
        # define graph connectivity based on communication radius
        x = self.feat_transformer(data.x, data.edge_attr, data.edge_index)
        h = self.feat_2_CBF(x)
        return h.reshape(-1, self.num_agents)


class CBFGNNLayer(gnn.MessagePassing):

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int):
        super(CBFGNNLayer, self).__init__(aggr='max')
        self.phi = MLP(in_channels=2 * node_dim + edge_dim, out_channels=phi_dim, hidden_layers=(64, 64))
        self.gamma = MLP(in_channels=phi_dim + node_dim, out_channels=output_dim, hidden_layers=(64, 64))

    def forward(self, x: Tensor, edge_attr: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor = None, edge_attr: Tensor = None) -> Tensor:
        info_ij = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.phi(info_ij)

    def update(self, aggr_out: Tensor, x: Tensor = None) -> Tensor:
        gamma_input = torch.cat([aggr_out, x], dim=1)
        return self.gamma(gamma_input)


# class GNNLayer(gnn.MessagePassing):
#
#     def __init__(self, feat_dim: int, phi_dim: int, aggr_dim: int, num_agents: int, state_dim: int):
#         super(GNNLayer, self).__init__(aggr='mean')
#         self.num_agents = num_agents
#         self.phi = MLP(in_channels=feat_dim + state_dim, out_channels=phi_dim, hidden_layers=(128, 128))
#         self.gamma = LSTM(input_size=aggr_dim + feat_dim, hidden_size=feat_dim, num_layers=2)
#
#     def forward(self, x, states):
#
#         # define graph connectivity based on communication radius
#         edge_index = build_comm_links(states, self.num_agents)
#
#         # run message passing
#         return self.propagate(edge_index, x=x, states=states)
#
#
#     def message(self, x_j, states_i, states_j):
#         info_ij = torch.cat([x_j, states_j - states_i], dim=1)
#         return self.phi(info_ij)
#
#     # def aggregate(self, phi_out, edge_index):
#     #     return phi_out
#
#     def update(self, aggr_out, x):
#         gamma_input = torch.cat([aggr_out, x], dim=1)
#         return self.gamma(gamma_input)