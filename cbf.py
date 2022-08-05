import torch
import torch.nn as nn
from core import *
from config import *


class CBF(nn.Module):

    def __init__(self, in_dim: int):
        super(CBF, self).__init__()

        layers = [
            nn.Conv1d(in_dim + 2, 64, (1,)), nn.ReLU(),
            nn.Conv1d(64, 128, (1,)), nn.ReLU(),
            nn.Conv1d(128, 64, (1,)), nn.ReLU(),
            nn.Conv1d(64, 1, (1,))
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, states):
        # preprocess input data
        x = torch.unsqueeze(states, dim=1) - torch.unsqueeze(states, dim=0)
        d_norm = torch.sqrt(
            torch.sum(torch.square(x[:, :, :2]) + 1e-4, dim=2))
        x = torch.concat(
            [x,
             torch.unsqueeze(torch.eye(x.size(dim=0)).type_as(x), dim=2),
             torch.unsqueeze(d_norm - DIST_MIN_THRES, dim=2)],
            dim=2)
        x, _ = remove_distant_agents(x, TOP_K)

        # build local observation mask
        dist = torch.sqrt(
            torch.sum(torch.square(x[:, :, :2]) + 1e-4, dim=2, keepdim=True))
        mask = torch.less_equal(dist, OBS_RADIUS)
        mask = mask.type_as(x).reshape(-1, mask.shape[2], mask.shape[1])

        # apply layers
        h = self.net(x.reshape(-1, x.shape[2], x.shape[1]))

        # apply mask
        h = h * mask

        return h, mask
