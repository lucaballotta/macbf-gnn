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


class Controller(nn.Module):

    def __init__(self, node_dim: int, edge_dim: int, phi_dim: int, num_agents: int, action_dim: int):
        super(Controller, self).__init__()

        self.num_agents = num_agents
        self.feat_transformer = gnn.Sequential('x, edge_attr, edge_index', [
            (ControllerGNNLayer(node_dim=node_dim, edge_dim=edge_dim, output_dim=64, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
            nn.ReLU(),
            (ControllerGNNLayer(node_dim=64, edge_dim=edge_dim, output_dim=64, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
        ])
        self.feat_2_action = MLP(in_channels=64, out_channels=action_dim, hidden_layers=(64, 64))

    def forward(self, data: Data) -> Tensor:
        x = self.feat_transformer(data.x, data.edge_attr, data.edge_index)
        actions = self.feat_2_action(x)
        
        return actions


class ControllerGNNLayer(gnn.MessagePassing):

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int):
        super(ControllerGNNLayer, self).__init__(aggr='max')
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