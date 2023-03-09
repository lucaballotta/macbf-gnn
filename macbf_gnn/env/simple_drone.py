import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import Tuple, Optional, Union
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms.radius_graph import RadiusGraph
from torch_geometric.utils import mask_to_index
from cvxpy import Variable, Expression

from .utils import lqr, plot_graph_3d
from .base import MultiAgentEnv


class SimpleDrone(MultiAgentEnv):

    def __init__(self, num_agents: int, device: torch.device, dt: float = 0.03, params: dict = None):
        super(SimpleDrone, self).__init__(num_agents, device, dt, params)

        # builder of the graph
        self._builder = RadiusGraph(self._params['comm_radius'], max_num_neighbors=self.num_agents)

        # parameters for the reference controller
        self._K = None
        self._goal = None

        # parameters for plotting
        self._xyz_min = np.array([0, 0, 0])
        self._xyz_max = np.ones(3) * self._params['area_size']

    @property
    def state_dim(self) -> int:
        return 8  # [x, y, z, vx, vy, vz, theta_x, theta_y]

    @property
    def node_dim(self) -> int:
        return 4

    @property
    def edge_dim(self) -> int:
        return 8

    @property
    def action_dim(self) -> int:
        return 3  # [ax, ay, az]

    @property
    def max_episode_steps(self) -> int:
        return 500

    @property
    def default_params(self) -> dict:
        return {
            'area_size': 1.,
            'drone_radius': 0.05,
            'comm_radius': 0.5,
            'dist2goal': 0.05
        }

    @property
    def _A(self) -> Tensor:
        A = torch.zeros(8, 8, dtype=torch.float, device=self.device)
        A[0, 3] = 1.
        A[1, 4] = 1.
        A[2, 5] = 1.
        A[3, 3] = -1.1
        A[4, 4] = -1.1
        A[5, 5] = -6.
        return A

    @property
    def _B(self) -> Tensor:
        B = torch.zeros(8, 3, dtype=torch.float, device=self.device)
        B[3, 0] = 1.1
        B[4, 1] = 1.1
        B[5, 2] = 6.
        return B

    def dynamics(self, x: Tensor, u: Union[Tensor, Expression]) -> Union[Tensor, Expression]:
        if isinstance(u, Expression):
            A = self._A.cpu().numpy()
            B = self._B.cpu().numpy()


        # xdot = Ax + Bu
        xdot = x @ self._A.t() + u @ self._B.t()
        return xdot

    def reset(self) -> Data:
        self._t = 0
        states = torch.zeros(self.num_agents, 8, device=self.device)
        goals = torch.zeros(self.num_agents, 8, device=self.device)

        # starting points are on the ground
        i = 0
        while i < self.num_agents:
            candidate = torch.rand(1, 3).type_as(states) * self._params['area_size']
            dist_min = torch.norm(states[:, :3] - candidate, dim=1).min()
            if dist_min <= self._params['drone_radius'] * 4:
                continue
            states[i, :3] = candidate
            i += 1

        i = 0
        while i < self.num_agents:
            candidate = torch.rand(1, 3).type_as(states) * self._params['area_size']
            dist_min = torch.norm(goals[:, :3] - candidate, dim=1).min()
            if dist_min <= self._params['drone_radius'] * 4:
                continue
            goals[i, :3] = candidate
            i += 1

        # record goals
        self._goal = goals

        # build graph
        data = Data(x=torch.zeros(self.num_agents, self.node_dim).type_as(states), pos=states[:, :3], states=states)
        data = self.add_communication_links(data)
        self._data = data

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
        data = Data(x=torch.zeros(self.num_agents, self.node_dim).type_as(state), pos=state[:, :3], states=state)
        self._data = self.add_communication_links(data)

        # the episode ends when reaching max_episode_steps or all the agents reach the goal
        time_up = self._t >= self.max_episode_steps
        reach = torch.less(
            torch.norm(self.data.states[:, :3] - self._goal[:, :3], dim=1), self._params['dist2goal']).all()
        done = time_up or reach

        # reward function
        reward = - (self.unsafe_mask(data).sum() * 100 / self.num_agents) - 1.0 + reach * 1000.

        safe = self.unsafe_mask(data).sum() == 0
        return self.data, float(reward), done, {'safe': safe}

    def forward_graph(self, data: Data, action: Tensor) -> Data:
        # calculate next state using dynamics
        action = action + self.u_ref(data)
        lower_lim, upper_lim = self.action_lim
        action = torch.clamp(action, lower_lim, upper_lim)
        state = self.forward(data.states, action)

        # construct the graph of the next step, retaining the connection
        data_next = Data(
            x=torch.zeros(state.shape[0], self.node_dim).type_as(state),
            edge_index=data.edge_index,
            edge_attr=state[data.edge_index[0]] - state[data.edge_index[1]],
            pos=state[:, :2],
            states=state
        )

        return data_next

    def edge_dynamics(self, data: Data, action: Variable) -> Variable:
        raise NotImplementedError

    def render(self, traj: Optional[Tuple[Data, ...]] = None, return_ax: bool = False, plot_edge: bool = True) -> \
            Union[Tuple[np.array, ...], np.array]:
        return_tuple = True
        if traj is None:
            data = self.data
            traj = (data,)
            return_tuple = False

        r = self._params['drone_radius']
        gif = []
        for data in traj:
            # 3D plot
            fig = plt.figure(figsize=(10, 10), dpi=80)
            ax = fig.add_subplot(projection='3d')

            # plot the drones and the communication network
            plot_graph_3d(ax, data, radius=r, color='#FF8C00', with_label=True,
                          plot_edge=plot_edge, alpha=0.3)

            # plot the goals
            goal_data = Data(pos=self._goal[:, :3])
            plot_graph_3d(ax, goal_data, radius=r, color='#3CB371', with_label=True,
                          plot_edge=False, alpha=0.3)

            # texts
            fontsize = 14
            collision_text = ax.text2D(0., 0.97, "", transform=ax.transAxes, fontsize=fontsize)
            unsafe_index = mask_to_index(self.unsafe_mask(data))
            collision_text.set_text(f'Collision: {unsafe_index.cpu().detach().numpy()}')

            # set axis limit
            ax.set_xlim(self._xyz_min[0], self._xyz_max[0])
            ax.set_ylim(self._xyz_min[1], self._xyz_max[1])
            ax.set_zlim(self._xyz_min[2], self._xyz_max[2])
            ax.set_aspect('equal')
            # plt.axis('off')

            if return_ax:
                return ax

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

    @property
    def state_lim(self) -> Tuple[Tensor, Tensor]:
        low_lim = torch.tensor(
            [self._xyz_min[0], self._xyz_min[1], self._xyz_min[2], -10, -10, -10, 0, 0], device=self.device)
        high_lim = torch.tensor(
            [self._xyz_max[0], self._xyz_max[1], self._xyz_max[2], 10, 10, 10, 0, 0], device=self.device)
        return low_lim, high_lim

    @property
    def action_lim(self) -> Tuple[Tensor, Tensor]:
        upper_limit = torch.ones(self.action_dim, device=self.device) * 10.0
        lower_limit = - upper_limit
        return lower_limit, upper_limit

    def u_ref(self, data: Data) -> Tensor:
        states = data.states.reshape(-1, self.num_agents, self.state_dim)
        diff = states - self._goal

        if self._K is None:
            # get used A, B, Q, R
            A = self._A.cpu().numpy() * self.dt + np.eye(self.state_dim)
            B = self._B.cpu().numpy() * self.dt
            A = A[:, :self.state_dim - 2][:self.state_dim - 2, :]
            B = B[:self.state_dim - 2, :]
            Q = np.eye(self.state_dim - 2)
            R = np.eye(self.action_dim)
            K_np = lqr(A, B, Q, R)
            K_np = np.concatenate((K_np, np.zeros((3, 2))), axis=1)
            self._K = torch.from_numpy(K_np).type_as(data.states)

        # feedback control
        action = - torch.einsum('us,bns->bnu', self._K, diff)
        return action.reshape(-1, self.action_dim)

    def safe_mask(self, data: Data) -> Tensor:
        mask = torch.empty(data.states.shape[0])
        for i in range(data.states.shape[0] // self.num_agents):
            state_diff = data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(1) - \
                         data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(0)
            pos_diff = state_diff[:, :, :3]
            dist = pos_diff.norm(dim=2)
            dist += torch.eye(dist.shape[0], device=self.device) * (2 * self._params['drone_radius'] + 1)
            safe = torch.greater(dist, 3 * self._params['drone_radius'])  # 3 is a hyperparameter
            mask[self.num_agents * i: self.num_agents * (i + 1)] = torch.min(safe, dim=1)[0]
        mask = mask.bool()

        return mask

    def unsafe_mask(self, data: Data) -> Tensor:
        mask = torch.empty(data.states.shape[0])
        for i in range(data.states.shape[0] // self.num_agents):
            state_diff = data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(1) - \
                         data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(0)
            pos_diff = state_diff[:, :, :3]
            dist = pos_diff.norm(dim=2)
            dist += torch.eye(dist.shape[0], device=self.device) * (2 * self._params['drone_radius'] + 1)
            collision = torch.less(dist, 2 * self._params['drone_radius'])
            mask[self.num_agents * i: self.num_agents * (i + 1)] = torch.max(collision, dim=1)[0]
        mask = mask.bool()

        return mask
