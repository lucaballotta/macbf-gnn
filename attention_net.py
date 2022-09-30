from torch import Tensor

from torch_geometric.nn import TransformerConv
from torch_geometric.nn import Sequential, MessagePassing, aggr


class AttentionNet(aggr.Aggregation):

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.aggregator = TransformerConv(in_channels=in_dim, out_channels=in_dim)


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.aggregator(x, edge_index)