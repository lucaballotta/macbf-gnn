import torch
from torch.nn import LSTM
import torch_geometric.nn as gnn
# from torch_geometric_temporal.nn.recurrent import GConvGRU
from core import *
from config import *


class GNN(nn.Module):
    


class GNNLayer(gnn.MessagePassing):

    def __init__(self, feat_dim: int, phi_dim: int, aggr_dim: int, num_agents: int, state_dim: int):
        super(GNNLayer, self).__init__()
        self.num_agents = num_agents
        self.phi = MLP(in_channels=feat_dim + state_dim, out_channels=phi_dim, hidden_layers=(20, 20))
        self.aggregator = gnn.TransformerConv(in_channels=phi_dim, out_channels=aggr_dim)
        self.gamma = LSTM(aggr_dim + feat_dim, 20, feat_dim)


    def forward(self, x, states):

        # define graph connectivity based on communication radius
        edge_index = build_comm_links(states, self.num_agents)

        # run message passing
        return self.propagate(edge_index, x=x, states=states)


    def message(self, x_j, states_i, states_j):
        info_ij = torch.cat([x_j, states_j - states_i], dim=1)
        return self.phi(info_ij)


    def aggregate(self, phi_out, edge_index):
        return self.aggregator(phi_out, edge_index)


    def update(self, aggr_out_i, x_i):
        gamma_input = torch.cat([aggr_out_i, x_i], dim=1)
        return self.gamma(gamma_input)