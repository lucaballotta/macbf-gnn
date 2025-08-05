import torch

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data

from macbf_gnn.env import MultiAgentEnv
from macbf_gnn.controller.qp_controller import ControllerQP


from .base import Algorithm


class QP(Algorithm):
    def __init__(
            self,
            env: MultiAgentEnv,
            num_agents: int,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            device: torch.device,
    ):
        super(QP, self).__init__(
            env=env,
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )
        self.actor = ControllerQP(
            num_agents=self.num_agents,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            action_dim=self.action_dim,
            env=env
        ).to(device)

    def step(self, data: Data) -> Tensor:
        raise NotImplementedError

    def is_update(self, step: int) -> bool:
        raise NotImplementedError

    def update(self, step: int, writer: SummaryWriter = None):
        raise NotImplementedError

    def save(self, save_dir: str):
        raise NotImplementedError

    def load(self, load_dir: str):
        raise NotImplementedError

    def act(self, data: Data) -> Tensor:
        with torch.no_grad():
            return self.actor(data.available_states)

    def apply(self, data: Data) -> Tensor:
        return self.act(data)
