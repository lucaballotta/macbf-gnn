from turtle import forward
import torch
from torch.nn import LSTM
import torch_geometric.nn as gnn
# from torch_geometric_temporal.nn.recurrent import GConvGRU
from core import *
from config import *


class GNN(nn.Module):

    def __init__(self, feat_dim: int, phi_dim: int, aggr_dim: int, num_agents: int, state_dim: int):
        super().__init__()
        self.GNN_layer1 = GNNLayer(feat_dim, phi_dim, aggr_dim, num_agents, state_dim)
        self.GNN_layer2 = GNNLayer(feat_dim, phi_dim, aggr_dim, num_agents, state_dim)
        self.relu = nn.ReLU()
        self.feat_transformer = gnn.Sequential('x, states', [
            (GNNLayer(feat_dim, phi_dim, aggr_dim, num_agents, state_dim), 'x, states -> x'),
            nn.ReLU(),
            (GNNLayer(feat_dim, phi_dim, aggr_dim, num_agents, state_dim), 'x, states -> x'),
            ])
        self.feat_2_CBF = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, 1)
            )


    def forward(self, x, states):
        x = self.GNN_layer1(x, states)
        x = self.relu(x)
        print('@@@@@@@@@@@@@@@@@')
        x = self.GNN_layer2(x, states)

        x = self.feat_transformer(x, states)
        h = self.feat_2_CBF(x)
        return x, h


class GNNLayer(gnn.MessagePassing):

    def __init__(self, feat_dim: int, phi_dim: int, aggr_dim: int, num_agents: int, state_dim: int):
        super(GNNLayer, self).__init__()
        self.num_agents = num_agents
        self.phi = MLP(in_channels=feat_dim + state_dim, out_channels=phi_dim, hidden_layers=(20, 20))
        self.aggregator = gnn.TransformerConv(in_channels=phi_dim, out_channels=aggr_dim)
        self.gamma = LSTM(input_size=aggr_dim + feat_dim, hidden_size=feat_dim, num_layers=2)


    def forward(self, x, states):

        # define graph connectivity based on communication radius
        edge_index = build_comm_links(states, self.num_agents)

        # run message passing
        return self.propagate(edge_index, x=x, states=states)


    def message(self, x_j, states_i, states_j):
        info_ij = torch.cat([x_j, states_j - states_i], dim=1)
        return self.phi(info_ij)


    def aggregate(self, phi_out, edge_index):
        out = self.aggregator(phi_out, edge_index)
        print('o',out.size())
        return out


    def update(self, aggr_out, x):
        gamma_input = torch.cat([aggr_out, x], dim=1)
        out = self.gamma(gamma_input)
        return out[0]