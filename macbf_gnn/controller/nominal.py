import torch

from torch import Tensor
from torch_geometric.data import Data

from .base import MultiAgentController


class NominalController(MultiAgentController):

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, action_dim: int):
        super(NominalController, self).__init__(
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim
        )

    def forward(self, data: Data) -> Tensor:
        return torch.zeros(data.num_nodes, self.action_dim)
