from turtle import forward
import torch
import torch.nn as nn
from core import *
from config import *

class CBF(nn.Module):

    def __init__(self, in_dim):
        layers_cbf = []
        layers_cbf.append(nn.Conv1d(in_dim, 64)) # input layer
        layers_cbf.append(nn.ReLU)
        layers_cbf.append(nn.Conv1d(64, 128))   # hidden layer
        layers_cbf.append(nn.ReLU)
        layers_cbf.append(nn.Conv1d(128, 64))   # hidden layer
        layers_cbf.append(nn.ReLU)
        layers_cbf.append(nn.Conv1d(64, 1))     # output layer
        self.cbf_net = nn.Sequential(*layers_cbf)


    def forward(self, states):

        # preprocess input data
        x = torch.unsqueeze(states, dim=1) - torch.unsqueeze(states, dim=0)
        d_norm = torch.sqrt(
            torch.sum(torch.square(x[:, :, :2]) + 1e-4, dim=2))
        x = torch.concat(
            [x,
            torch.unsqueeze(torch.eye(x.size(dim=0)), dim=2),
            torch.unsqueeze(d_norm - DIST_MIN_THRES, dim=2)],
            axis=2)
        x, _ = remove_distant_agents(x, TOP_K)

        # build local observation mask
        dist = torch.sqrt(
            torch.sum(torch.square(x[:, :, :2]) + 1e-4, dim=2, keepdim=True))
        mask = torch.less_equal(dist, OBS_RADIUS)
        mask.type(torch.float32)

        # apply layers
        h = self.cbf_net(x)

        # apply mask
        h = h * mask

        return h


    def parameters(self):
        return self.cbf_net.parameters()