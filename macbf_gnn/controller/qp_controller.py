import torch
import cvxpy as cp

from torch_geometric.data import Data
from torch import Tensor

from .base import MultiAgentController
from macbf_gnn.env import MultiAgentEnv


class ControllerQP(MultiAgentController):

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, action_dim: int, env: MultiAgentEnv):
        super(ControllerQP, self).__init__(
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim
        )
        self._env = env
        
    def forward(self, data: Data) -> Tensor:
        action = cp.Variable((self.num_agents, self.action_dim))
        obj = cp.Minimize(cp.sum_squares(action))
        data_next = self._env.forward_graph(data, action)
        constraints = []
        for agent in range(self.num_agents):
            for other_agent in range(self.num_agents):
                if agent == other_agent:
                    continue
                
                cbf_agent_pair = (data.pos[agent] - data.pos[other_agent])**2 - (2 * self._env._params['car_radius'])**2
                cbf_next_agent_pair = (data_next.pos[agent] - data_next.pos[other_agent])**2 - (2 * self._env._params['car_radius'])**2
                cbf_agent_pair_dot = (cbf_next_agent_pair - cbf_agent_pair) / self._env.dt
                agent_pair_constraint = cp.Constraint([cbf_agent_pair_dot + self._env._params['alpha'] * cbf_agent_pair >= 0])
                constraints.extend(agent_pair_constraint)
                
        prob = cp.Problem(obj, constraints)
        _ = prob.solve()
        
        return action.value()