import torch.nn as nn

from torch_geometric.nn import Sequential
from torch_geometric.nn.conv.transformer_conv import TransformerConv
from torch import Tensor, cat
from torch_geometric.data import Data

from macbf_gnn.nn.gnn import ControllerGNNLayer
from macbf_gnn.nn.mlp import MLP

from .base import MultiAgentController


class ControllerQP(MultiAgentController):

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, action_dim: int):
        super(ControllerQP, self).__init__(
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim
        )
        self.cbf = 2
        
    def forward(self, data: Data) -> Tensor:
        h = self.cbf(data)
        actions = []
        for i_node in range(self.num_agents):
            action = cp.Variable(self.action_dim)
            obj = cp.Minimize(cp.sum_squares(action))
            graph_next = self._env.forward_agent(data, action, i_node)
            h_next = self.cbf(graph_next)
            h_dot = (h_next - h) / self._env.dt
            max_val_h_dot = torch.mean(torch.relu(-h_dot - self.params['alpha'] * h))
            constraints = [max_val_h_dot <= 0]
            prob = cp.Problem(obj, constraints)
            result = prob.solve()
            actions.append(torch.tensor(action.value).type_as(data.states))
            
        return torch.cat(actions)