import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

from typing import Tuple, Optional, Union
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms.radius_graph import RadiusGraph
from torch_geometric.utils import to_networkx, mask_to_index

from .utils import lqr
from .base import MultiAgentEnv


class SimpleCar(MultiAgentEnv):

    def __init__(self, num_agents: int, device: torch.device, dt: float = 0.05, params: dict = None):
        super(SimpleCar, self).__init__(num_agents, device, dt, params)

        # builder of the graph
        self._builder = RadiusGraph(self._params['comm_radius'], max_num_neighbors=self.num_agents)

        # parameters for the reference controller
        self._K = None
        self._goal = None

        # parameters for plotting
        self._xy_min = None
        self._xy_max = None

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def node_dim(self) -> int:
        return 4

    @property
    def edge_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def max_episode_steps(self) -> int:
        return 500

    @property
    def default_params(self) -> dict:
        return {
            'm': 1.0,  # mass of the car
            'comm_radius': 1.0,  # communication radius
            'car_radius': 0.05,  # radius of the cars
            'dist2goal': 0.03  # goal reaching threshold
        }

    def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        return torch.cat([x[:, 2:], u], dim=1)

    def reset(self) -> Data:
        self._t = 0
        side_length = np.sqrt(max(1.0, self.num_agents / 8.0))
        states = torch.zeros(self.num_agents, 2, device=self.device)
        goals = torch.zeros(self.num_agents, 2, device=self.device)

        # randomly generate positions of agents
        i = 0
        while i < self.num_agents:
            candidate = torch.rand(2, device=self.device) * side_length
            dist_min = torch.norm(states - candidate, dim=1).min()
            if dist_min <= self._params['car_radius'] * 2:
                continue
            states[i] = candidate
            i += 1

        # randomly generate goals of agents
        i = 0
        while i < self.num_agents:
            candidate = (torch.rand(2, device=self.device) - 0.5) + states[i]
            dist_min = torch.norm(goals - candidate, dim=1).min()
            if dist_min <= self._params['car_radius'] * 2:
                continue
            goals[i] = candidate
            i += 1

        # add velocity
        states = torch.cat([states, torch.zeros(self.num_agents, 2, device=self.device)], dim=1)

        # record goals
        self._goal = goals

        # build graph
        data = Data(x=torch.zeros_like(states), pos=states[:, :2], states=states)
        data = self.add_communication_links(data)
        self._data = data

        # set parameters for plotting
        points = torch.cat([states[:, :2], goals], dim=0).cpu().detach().numpy()
        xy_min = np.min(points, axis=0) - self._params['car_radius'] * 5
        xy_max = np.max(points, axis=0) + self._params['car_radius'] * 5
        max_interval = (xy_max - xy_min).max()
        self._xy_min = xy_min - 0.5 * (max_interval - (xy_max - xy_min))
        self._xy_max = xy_max + 0.5 * (max_interval - (xy_max - xy_min))

        return data

    def step(self, action: Tensor) -> Tuple[Data, float, bool, dict]:
        self._t += 1

        # calculate next state using dynamics
        action = action + self.u_ref(self._data)
        lower_lim, upper_lim = self.action_lim
        action = torch.clamp(action, lower_lim, upper_lim)
        with torch.no_grad():
            state = self.forward(self.state, action)

        # construct graph using the new states
        data = Data(x=torch.zeros_like(state), pos=state[:, :2], states=state)
        self._data = self.add_communication_links(data)

        # the episode ends when reaching max_episode_steps or all the agents reach the goal
        time_up = self._t >= self.max_episode_steps
        reach = torch.less(torch.norm(self.data.states[:, :2] - self._goal, dim=1), self._params['dist2goal']).all()
        done = time_up or reach

        # reward function
        reward = - (self.unsafe_mask(data).sum() * 10 / self.num_agents) - 1.0 + reach * 100.

        return self.data, float(reward), done, {}

    def forward_graph(self, data: Data, action: Tensor) -> Data:
        # calculate next state using dynamics
        action = action + self.u_ref(data)
        lower_lim, upper_lim = self.action_lim
        action = torch.clamp(action, lower_lim, upper_lim)
        state = self.forward(data.states, action)

        # construct the graph of the next step, retaining the connection
        data_next = Data(
            x=torch.zeros_like(state),
            edge_index=data.edge_index,
            edge_attr=state[data.edge_index[0]] - state[data.edge_index[1]],
            pos=state[:, :2],
            states=state
        )

        return data_next

    def render(self, traj: Optional[Tuple[Data, ...]] = None) -> Union[Tuple[np.array, ...], np.array]:
        return_tuple = True
        if traj is None:
            data = self.data
            traj = (data,)
            return_tuple = False

        gif = []
        for data in traj:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)

            # calculate car size in display
            r = ax.transData.transform([self._params['car_radius'], 0])[0] - ax.transData.transform([0, 0])[0]
            s = 4 * r ** 2

            # plot the cars and the communication network
            graph = to_networkx(data)
            pos = data.pos.cpu().detach().numpy()
            nx.draw(graph, pos=pos, ax=ax, with_labels=True, node_size=s,
                    node_color='#FF8C00', linewidths=0., node_shape='.', alpha=0.8)

            # plot the goals
            goal_graph = nx.Graph()
            goal_graph.add_nodes_from(range(self._goal.shape[0]))
            goal_pos = self._goal[:, :2].cpu().numpy()
            nx.draw(goal_graph, pos=goal_pos, ax=ax, with_labels=True,
                    node_size=s, node_color='#3CB371', alpha=0.5, linewidths=0., node_shape='.')

            # texts
            fontsize = 14
            collision_text = ax.text(0., 0.97, "", transform=ax.transAxes, fontsize=fontsize)
            unsafe_index = mask_to_index(self.unsafe_mask(data))
            collision_text.set_text(f'Collision: {unsafe_index.cpu().detach().numpy()}')

            # set axis limit
            ax.set_xlim(self._xy_min[0], self._xy_max[0])
            ax.set_ylim(self._xy_min[1], self._xy_max[1])

            # convert to numpy array
            fig.canvas.draw()
            fig_np = np.frombuffer(
                fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
            gif.append(fig_np)
            plt.close()

        if return_tuple:
            return tuple(gif)
        else:
            return gif[0]

    def add_communication_links(self, data: Data) -> Data:
        data = self._builder(data)
        data.edge_attr = data.states[data.edge_index[0]] - data.states[data.edge_index[1]]
        return data

    # @property
    # def state_lim(self) -> Tuple[Tensor, Tensor]:
    #     pass
    #
    # @property
    # def node_lim(self) -> Tuple[Tensor, Tensor]:
    #     return (
    #         torch.zeros(4),
    #         torch.zeros(4)
    #     )
    #
    # @property
    # def edge_lim(self) -> Tuple[Tensor, Tensor]:
    #     pass

    @property
    def action_lim(self) -> Tuple[Tensor, Tensor]:
        upper_limit = torch.ones(2, device=self.device) * 5.0
        lower_limit = - upper_limit
        return lower_limit, upper_limit

    def u_ref(self, data: Data) -> Tensor:
        goal = torch.cat([self._goal, torch.zeros_like(self._goal)], dim=1)
        states = data.states.reshape(-1, self.num_agents, self.state_dim)
        diff = (states - goal)

        if self._K is None:
            # calculate the LQR controller
            A = np.array([[0., 0., 1., 0.],
                          [0., 0., 0., 1.],
                          [0., 0., 0., 0.],
                          [0., 0., 0., 0.]]) * self.dt + np.eye(self.state_dim)
            B = np.array([[0., 0.],
                          [0., 0.],
                          [1., 0.],
                          [0., 1.]]) * self.dt
            Q = np.eye(self.state_dim)
            R = np.eye(self.action_dim)
            K_np = lqr(A, B, Q, R)
            self._K = torch.from_numpy(K_np).type_as(data.states)

        # feedback control
        action = - torch.einsum('us,bns->bnu', self._K, diff)
        return action.reshape(-1, self.action_dim)

    def safe_mask(self, data: Data) -> Tensor:
        return torch.logical_not(self.unsafe_mask(data))

    def unsafe_mask(self, data: Data) -> Tensor:  # todo: bug exists for batched data
        state_diff = data.states.unsqueeze(1) - data.states.unsqueeze(0)
        pos_diff = state_diff[:, :, :2]
        dist = pos_diff.norm(dim=2)
        dist += torch.eye(dist.shape[0], device=self.device) * (2 * self._params['car_radius'] + 1)
        collision = torch.less(dist, 2 * self._params['car_radius'])
        mask = torch.max(collision, dim=1)[0]
        return mask
