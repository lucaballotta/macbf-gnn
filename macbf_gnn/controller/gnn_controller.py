import torch.nn as nn

from torch_geometric.nn import Sequential
from torch_geometric.nn.conv.transformer_conv import TransformerConv
from torch import Tensor
from torch_geometric.data import Data

from macbf_gnn.nn.gnn import ControllerGNNLayer
from macbf_gnn.nn.mlp import MLP

from .base import MultiAgentController


class GNNController(MultiAgentController):

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, phi_dim: int, action_dim: int):
        super(GNNController, self).__init__(
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim
        )

        self.feat_transformer = Sequential('x, edge_attr, edge_index', [
            (ControllerGNNLayer(node_dim=node_dim, edge_dim=edge_dim, output_dim=512, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
            nn.ReLU(),
            (ControllerGNNLayer(node_dim=512, edge_dim=edge_dim, output_dim=128, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
            nn.ReLU(),
            (ControllerGNNLayer(node_dim=128, edge_dim=edge_dim, output_dim=64, phi_dim=phi_dim),
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
        self.feat_2_action = MLP(in_channels=64, out_channels=action_dim, hidden_layers=(64, 64))

    def forward(self, data: Data) -> Tensor:
        """
        Get the control actions for the input states.

        Parameters
        ----------
        data: Data
            batched data using Batch.from_data_list().

        Returns
        -------
        actions: (bs x n, action_dim)
            control actions for all agents
        """
        x = self.feat_transformer(data.x, data.edge_attr, data.edge_index)
        actions = self.feat_2_action(x)

        return actions
