import torch

from torch_geometric.nn.conv.message_passing import MessagePassing
from torch import Tensor, cat
from torch_sparse import SparseTensor

from .utils import AttentionAggregation
from .mlp import MLP


class CBFGNNLayer(MessagePassing):

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int):
        super(CBFGNNLayer, self).__init__(aggr=AttentionAggregation(
            gate_nn=MLP(in_channels=phi_dim, out_channels=phi_dim, hidden_layers=(32,))))
        self.phi = MLP(in_channels=2 * node_dim + edge_dim, out_channels=phi_dim, hidden_layers=(64, 64))
        self.gamma = MLP(in_channels=phi_dim + node_dim, out_channels=output_dim, hidden_layers=(64, 64))

    def forward(self, x: Tensor, edge_attr: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor = None, edge_attr: Tensor = None) -> Tensor:
        info_ij = cat([x_i, x_j, edge_attr], dim=1)
        return self.phi(info_ij)

    def update(self, aggr_out: Tensor, x: Tensor = None) -> Tensor:
        gamma_input = cat([aggr_out, x], dim=1)
        return self.gamma(gamma_input)

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError

    def edge_update(self) -> Tensor:
        raise NotImplementedError


class ControllerGNNLayer(MessagePassing):

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int):
        super(ControllerGNNLayer, self).__init__(aggr=AttentionAggregation(
            gate_nn=MLP(in_channels=phi_dim, out_channels=phi_dim, hidden_layers=(32,))))
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

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError

    def edge_update(self) -> Tensor:
        raise NotImplementedError
