import cvxpy as cp

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
        self.alpha = 1.
        
    def forward(self, state: Tensor) -> Tensor:
        action = cp.Variable((self.num_agents, self.action_dim))
        obj = cp.Minimize(cp.sum_squares(action))
        state_next = self._env.forward(state, action)
        constraints = []
        for agent in range(self.num_agents):
            for other_agent in range(self.num_agents):
                if agent == other_agent:
                    continue
                
                cbf_agent_pair = (state[agent] - state[other_agent])**2 - (2 * self._env._params['car_radius'])**2
                cbf_next_agent_pair = (state_next[agent] - state_next[other_agent])**2 - (2 * self._env._params['car_radius'])**2
                cbf_agent_pair_dot = (cbf_next_agent_pair - cbf_agent_pair) / self._env.dt
                agent_pair_constraint = cp.Constraint([cbf_agent_pair_dot + self.alpha * cbf_agent_pair >= 0])
                constraints.append(agent_pair_constraint)
                
        prob = cp.Problem(obj, constraints)
        _ = prob.solve()
        
        return action.value()