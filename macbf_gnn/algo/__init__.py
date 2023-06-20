import torch

from typing import Optional

from .base import Algorithm
from .cbf_gnn import MACBFGNN
from .nominal import Nominal
from ..env import MultiAgentEnv


def make_algo(
        algo: str,
        env: MultiAgentEnv,
        num_agents: int,
        node_dim: int,
        edge_dim: int,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        batch_size: int = 512,
        hyperparams: Optional[dict] = None
) -> Algorithm:
    if algo == 'nominal':
        return Nominal(
            env, num_agents, node_dim, edge_dim, action_dim, device
        )
    if algo == 'macbfgnn':
        return MACBFGNN(
            env, num_agents, node_dim, edge_dim, state_dim, action_dim, device, batch_size, hyperparams
        )
    else:
        raise NotImplementedError('Unknown Algorithm!')
