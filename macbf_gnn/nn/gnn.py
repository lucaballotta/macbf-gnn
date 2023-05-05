import torch
import numpy as np

from torch_geometric.nn.conv.message_passing import MessagePassing
from torch import Tensor, cat
from torch_sparse import SparseTensor
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torch_geometric.data import Data
from torch_geometric.utils import softmax
from torch.nn import LSTM
from torch.nn.utils.rnn import pad_packed_sequence

# from .utils import AttentionAggregation
from .mlp import MLP


class CBFGNNLayer(MessagePassing):

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int):
        super(CBFGNNLayer, self).__init__(aggr=AttentionalAggregation(
            gate_nn=MLP(in_channels=phi_dim, out_channels=1, hidden_layers=(128, 128), limit_lip=False)
        ))
        self.predictor = LSTM(
            input_size=edge_dim + 1, hidden_size=edge_dim, num_layers=4, batch_first=True
        )
        self.phi = MLP(
            in_channels=2 * node_dim + edge_dim, out_channels=phi_dim, hidden_layers=(2048, 2048), limit_lip=True
        )
        self.gamma = MLP(
            in_channels=phi_dim + node_dim, out_channels=output_dim, hidden_layers=(2048, 2048), limit_lip=True
        )

    def forward(self, x: Tensor, edge_attr: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor = None, edge_attr: Tensor = None) -> Tensor:
        output, _ = self.predictor(edge_attr)
        padded_output = pad_packed_sequence(output, batch_first=True)
        pred = padded_output[0]
        lengths = padded_output[1]
        pred_last = pred[np.arange(len(pred)), lengths - 1]
        info_ij = cat([x_i, x_j, pred_last], dim=1)
        return self.phi(info_ij)

    def update(self, aggr_out: Tensor, x: Tensor = None) -> Tensor:
        gamma_input = cat([aggr_out, x], dim=1)
        return self.gamma(gamma_input)

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError

    def edge_update(self) -> Tensor:
        raise NotImplementedError

    def attention(self, data: Data):
        kwargs = {'x': data.x, 'edge_attr': data.edge_attr}
        size = self.__check_input__(data.edge_index, None)
        coll_dict = self.__collect__(self.__user_args__, data.edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        out = self.message(**msg_kwargs)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        gate = self.aggr_module.gate_nn(out)
        attention = softmax(gate, aggr_kwargs['index'], aggr_kwargs['ptr'], aggr_kwargs['dim_size'], dim=-2)
        return attention


class ControllerGNNLayer(MessagePassing):

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int):
        super(ControllerGNNLayer, self).__init__(aggr=AttentionalAggregation(
            gate_nn=MLP(in_channels=phi_dim, out_channels=1, hidden_layers=(128, 128))
        ))
        self.predictor = LSTM(
            input_size=edge_dim + 1, hidden_size=edge_dim, num_layers=4, batch_first=True
        )
        self.phi = MLP(
            in_channels=2 * node_dim + edge_dim, out_channels=phi_dim, hidden_layers=(2048, 2048)
        )
        self.gamma = MLP(
            in_channels=phi_dim + node_dim, out_channels=output_dim, hidden_layers=(2048, 2048)
        )

    def forward(self, x: Tensor, edge_attr: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor = None, edge_attr: Tensor = None) -> Tensor:
        output, _ = self.predictor(edge_attr)
        padded_output = pad_packed_sequence(output, batch_first=True)
        pred = padded_output[0]
        lengths = padded_output[1]
        pred_last = pred[np.arange(len(pred)), lengths - 1]
        info_ij = cat([x_i, x_j, pred_last], dim=1)
        return self.phi(info_ij)

    def update(self, aggr_out: Tensor, x: Tensor = None) -> Tensor:
        gamma_input = torch.cat([aggr_out, x], dim=1)
        return self.gamma(gamma_input)

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError

    def edge_update(self) -> Tensor:
        raise NotImplementedError
