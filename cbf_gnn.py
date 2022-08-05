import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from core import *
from config import *


class CBF(nn.Module):

    def __init__(self, in_dim: int, num_agents: int):
        super(CBF, self).__init__()
        self.net = gnn.Sequential('states, edges', [
            gnn.GCNConv(in_dim, 64),
            nn.ReLU(),
            gnn.GCNConv(64, 64),
            nn.ReLU(),
            nn.Conv1d(64, 1, (1,))
            ])
        self.num_agents = num_agents

    def forward(self, states):
        
        # compute mask for inter-agent distances
        states_diff = torch.unsqueeze(states, dim=1) - torch.unsqueeze(states, dim=0)
        dist = torch.norm(states_diff[:, :, :2], dim=2, keepdim=True)
        mask = torch.squeeze(torch.less_equal(dist, OBS_RADIUS))
        
        # build sensing graph
        edge_sources = torch.tensor([], dtype=torch.int64)
        edge_sinks = torch.tensor([], dtype=torch.int64)
        
        for agent in range(self.num_agents):
            neighs = torch.cat(torch.where(mask[agent]), dim=0)
            neighs = neighs[neighs!=agent]
            edge_sources = torch.cat([edge_sources, torch.ones_like(neighs) * agent])
            edge_sinks = torch.cat([edge_sinks, neighs])
            
        edges = edges.tensor([edge_sources.tolist(), edge_sinks.tolist()])

        # apply GNN
        h = self.net(states, edges)

        return h, mask
