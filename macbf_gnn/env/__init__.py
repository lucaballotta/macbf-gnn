import torch

from .base import MultiAgentEnv
from .simple_car import SimpleCar
from .simple_drone import SimpleDrone
from .dubins_car import DubinsCar
# from .drone2d import Drone2D


def make_env(env: str, num_agents: int, device: torch.device, dt: float = 0.03, params: dict = None, delay_aware: bool = True):
    if env == 'SimpleCar':
        return SimpleCar(num_agents, device, dt, params, delay_aware)
    elif env == 'DubinsCar':
        return DubinsCar(num_agents, device, dt, params, delay_aware)
    elif env == 'SimpleDrone':
        return SimpleDrone(num_agents, device, dt, params, delay_aware)
    else:
        raise NotImplementedError('Env name not supported!')
