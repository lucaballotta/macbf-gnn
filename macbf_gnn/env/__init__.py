import torch

from .base import MultiAgentEnv
from .simple_car import SimpleCar
from .simple_drone import SimpleDrone


def make_env(env: str, num_agents: int, device: torch.device, dt: float = 0.03, params: dict = None):
    if env == 'SimpleCar':
        return SimpleCar(num_agents, device, dt, params)
    elif env == 'SimpleDrone':
        return SimpleDrone(num_agents, device, dt, params)
    else:
        raise NotImplementedError('Env name not supported!')
