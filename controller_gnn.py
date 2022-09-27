import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from torch_geometric.data import Data
from torch import Tensor
from torch_scatter import scatter, scatter_add, scatter_softmax
from torch.nn import LSTM

from utils import *
from config import *
from MLP import MLP


class ControllerGNN(nn.Module):

    def __init__(self, state_dim: int, phi_dim: int, num_agents: int, goal_dim: int):
        super(ControllerGNN, self).__init__()

        self.num_agents = num_agents
        self.feat_transformer = gnn.Sequential('x, edge_attr, edge_index', [
            (ControllerGNNLayer(node_dim=state_dim, edge_dim=state_dim, output_dim=64, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
            nn.ReLU(),
            (ControllerGNNLayer(node_dim=64, edge_dim=state_dim, output_dim=64, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
        ])
        input_dim = 64 + num_agents * goal_dim
        self.feat_2_gains = MLP(in_channels=input_dim, out_channels=4, hidden_layers=(64, 64))

    def forward(self, data: Data) -> Tensor:
        feat = self.feat_transformer(data.x, data.edge_attr, data.edge_index)
        input = torch.cat([feat, data.goal], dim=0)
        out_gains = self.feat_2_gains(input)

        # extract control actions
        out_gains = 2.0 * torch.sigmoid(out_gains) + 0.2
        k_1, k_2, k_3, k_4 = torch.split(out_gains, 1, dim=1)
        zeros = torch.zeros_like(k_1)
        gain_x = -torch.concat([k_1, zeros, k_2, zeros], dim=1)
        gain_y = -torch.concat([zeros, k_3, zeros, k_4], dim=1)
        state = torch.concat([states[:, :2] - goals, states[:, 2:]], dim=1)
        a_x = torch.sum(state * gain_x, dim=1, keepdim=True)
        a_y = torch.sum(state * gain_y, dim=1, keepdim=True)
        a = torch.concat([a_x, a_y], dim=1)
        
        return a


class ControllerGNNLayer(gnn.MessagePassing):

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int):
        super(CBFGNNLayer, self).__init__(aggr='max')
        self.phi = MLP(in_channels=2 * node_dim + edge_dim, out_channels=phi_dim, hidden_layers=(64, 64))
        self.gamma = MLP(in_channels=phi_dim + node_dim, out_channels=output_dim, hidden_layers=(64, 64))

    def forward(self, x: Tensor, edge_attr: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor = None, edge_attr: Tensor = None) -> Tensor:
        info_ij = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.phi(info_ij)
    
#     # def aggregate(self, phi_out, edge_index):
#     #     return phi_out

    def update(self, aggr_out: Tensor, x: Tensor = None) -> Tensor:
        gamma_input = torch.cat([aggr_out, x], dim=1)
        return self.gamma(gamma_input)