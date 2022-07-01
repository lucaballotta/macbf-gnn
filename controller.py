import itertools
from turtle import forward
from typing_extensions import Self
import torch
import torch.nn as nn
from core import *
from config import *

class Controller(nn.Module):

    def __init__(self, in_dim):

        # Centralized controller
        layers_centr = []
        layers_centr.append(nn.Conv1d(in_dim, 64)) # input layer
        layers_centr.append(nn.ReLU)
        layers_centr.append(nn.Conv1d(64, 128))   # hidden layer
        layers_centr.append(nn.ReLU)
        self.controller_centr_net = nn.Sequential(*layers_centr)

        # Decentralized controller
        layers_dec = []
        layers_dec.append(nn.Linear(128+4, 64)) # hidden layer
        layers_dec.append(nn.ReLU)
        layers_dec.append(nn.Linear(64, 128))   # hidden layer
        layers_dec.append(nn.ReLU)
        layers_dec.append(nn.Linear(128, 64))   # hidden layer
        layers_dec.append(nn.ReLU)
        layers_dec.append(nn.Linear(64, 4))     # output layer
        self.controller_dec_net = nn.Sequential(*layers_dec)

    def forward(self, states, goals):

        # preprocess input data
        x = torch.unsqueeze(states, dim=1) - torch.unsqueeze(states, dim=0)
        x = torch.concat(
            [x, torch.unsqueeze(torch.eye(x.size(dim=0)), 2)], dim=2)
        x, _ = remove_distant_agents(x, TOP_K)

        # build local observation mask
        dist = torch.norm(x[:, :, :2], dim=2, keepdim=True)
        mask = torch.less(dist, OBS_RADIUS)
        mask.type(torch.float32)

        # apply centralized layers
        x = self.controller_centr_net(x)

        # apply mask and max-pooling
        x_local = torch.max(x * mask, dim=1)
        x_local = torch.concat(
            [x_local, states[:, :2] - goals, states[:, 2:]], dim=1)

        # apply decentralized layers
        x_out = self.controller_dec_net(x_local)

        # extract control actions
        x_out = 2.0 * torch.sigmoid(x_out) + 0.2
        k_1, k_2, k_3, k_4 = torch.split(x_out, 4, dim=1)
        zeros = torch.zeros_like(k_1)
        gain_x = -torch.concat([k_1, zeros, k_2, zeros], dim=1)
        gain_y = -torch.concat([zeros, k_3, zeros, k_4], dim=1)
        state = torch.concat([states[:, :2] - goals, states[:, 2:]], dim=1)
        a_x = torch.sum(state * gain_x, dim=1, keepdim=True)
        a_y = torch.sum(state * gain_y, dim=1, keepdim=True)
        a = torch.concat([a_x, a_y], dim=1)
        
        return a


    def parameters(self):
        return itertools.chain(self.controller_centr_net.parameters(), self.controller_dec_net.parameters())    
