import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import Tuple, Optional, Union, List
from collections import deque
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms.radius_graph import RadiusGraph
from torch_geometric.utils import mask_to_index
from torch.nn.utils.rnn import pack_sequence
from cvxpy import Expression
from scipy.stats import poisson
from torch.nn.utils.rnn import pack_sequence, PackedSequence

from .utils import plot_graph
from .base import Agent, MultiAgentEnv


class DubinsCar(MultiAgentEnv):
    """
    State: [x, y, theta, v_x, v_y]
    """

    def __init__(self, num_agents: int, device: torch.device, dt: float = 0.03, params: dict = None, delay_aware: bool = True):
        super(DubinsCar, self).__init__(num_agents, device, dt, params, delay_aware)
        self._ref_path = None

        # states and actions along trajectory
        self._states = deque(maxlen=self._params['buffer_size'])
        self._actions = deque(maxlen=self._params['buffer_size'])  # actions that generate the states in self._states from previous states
        
        # cars
        car = Agent(buffer_size=self._params['buffer_size'], max_age=self._params['max_age'], device=self.device)
        self._cars = [car.copy() for _ in range(self.num_agents)]

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
        return 5

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def state_lim(self) -> Tuple[Tensor, Tensor]:
        low_lim = torch.tensor(
            [self._xy_min[0], self._xy_min[1], -10, -10],
            device=self.device)
        high_lim = torch.tensor(
            [self._xy_max[0], self._xy_max[1], 10, 10],
            device=self.device)
        return low_lim, high_lim

    @property
    def action_lim(self) -> Tuple[Tensor, Tensor]:
        upper_limit = torch.ones(2, device=self.device) * 10.
        lower_limit = - upper_limit
        return lower_limit, upper_limit
    
    @property
    def max_episode_steps(self) -> int:
        return 500

    @property
    def default_params(self) -> dict:
        return {
            'max_distance': 1.5,  # maximum moving distance to goal
            'area_size': 3.0,
            'car_radius': 0.05,
            'dist2goal': 0.05,
            'comm_radius': 1.0,
            'buffer_size': 5,  # max number of transmissions (tx delays) stored by cars
            'max_age': 5,  # max age of data stored by cars (older are discarded)
            'poisson_coeff': .2
        }
    

    def dynamics(self, x: Tensor, u: Union[Tensor, Expression]) -> Union[Tensor, Expression]:
        if isinstance(u, Expression):
            raise NotImplementedError
        
        else:
            xdot = torch.zeros_like(x)
            xdot[:, 0] = x[:, 3] * torch.cos(x[:, 2])
            xdot[:, 1] = x[:, 3] * torch.sin(x[:, 2])
            xdot[:, 2] = u[:, 0] * 10
            xdot[:, 3] = u[:, 1]
            if x.shape[0] == self.num_agents:
                reach = torch.less(torch.norm(x[:, :2] - self._goal[:, :2], dim=1), self._params['dist2goal'])
                return xdot * torch.logical_not(reach).unsqueeze(1).repeat(1, self.state_dim)
            else:
                return xdot
            

    def reset(self) -> Data:
        self._t = 0
        side_length = self._params['area_size']
        states = torch.zeros(self.num_agents, 2, device=self.device)
        goals = torch.zeros(self.num_agents, 2, device=self.device)

        if self._mode == 'train' or self._mode == 'test':
            
            # randomly generate positions of agents
            i = 0
            while i < self.num_agents:
                candidate = torch.rand(2, device=self.device) * side_length
                dist_min = torch.norm(states - candidate, dim=1).min()
                if dist_min <= self._params['car_radius'] * 4:
                    continue
                states[i] = candidate
                i += 1

            # randomly generate goals of agents
            i = 0
            while i < self.num_agents:
                candidate = torch.rand(2, device=self.device) * side_length
                dist_min = torch.norm(goals - candidate, dim=1).min()
                if dist_min <= self._params['car_radius'] * 5:
                    continue
                goals[i] = candidate
                i += 1
        else:
            raise ValueError('Reset environment: unknown type of mode!')

        # add velocity and heading
        states = torch.cat([states, torch.zeros(self.num_agents, 2, device=self.device)], dim=1)
        states[:, 2] = torch.rand_like(states[:, 3]) * 2 * torch.pi - torch.pi

        # record goals
        goals = torch.cat([goals, torch.zeros(self.num_agents, 2, device=self.device)], dim=1)
        goals[:, 2] = torch.rand_like(goals[:, 3]) * 2 * torch.pi - torch.pi
        self._goal = goals

        # store states
        states_store = torch.cat([states[:, :3],
                                  (states[:, 3] * torch.cos(states[:, 2])).unsqueeze(1),
                                  (states[:, 3] * torch.sin(states[:, 2])).unsqueeze(1)], dim=1)
        self._states.append(states_store)

        # initialize actions to zero
        self._actions.append(torch.zeros(self.num_agents, self.action_dim, device=self.device))
        
        # reset agents' data
        [car.reset_data() for car in self._cars]

        # build graph
        data = Data(x=torch.zeros(self.num_agents, self.node_dim).type_as(states), pos=states[:, :2], states=states)
        neigh_sizes = self.add_communication_links(data)

        # simulate transmission of delayed data among cars
        self.transmit_data(neigh_sizes)
        
        # simulate data reception at cars
        self.receive_data(data)
        
        # remove neighbors with age older than maximum allowed
        [car.remove_old_data() for car in self._cars]
        
        # store current edge attributes with received delayed data
        self._data = self.add_edge_attributes(data)

        # set parameters for plotting
        points = torch.cat([states[:, :2], goals[:, :2]], dim=0).cpu().detach().numpy()
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
            self._actions.append(action)
            state = self.forward(self.state, action)
            state_store = torch.cat([state[:, :3],
                                     (state[:, 3] * torch.cos(state[:, 2])).unsqueeze(1),
                                     (state[:, 3] * torch.sin(state[:, 2])).unsqueeze(1)], dim=1)
            self._states.append(state_store)
        
        # construct graph using the new states
        data = Data(x=torch.zeros(self.num_agents, self.node_dim).type_as(state), pos=state[:, :2], states=state)
        neigh_sizes = self.add_communication_links(data)

        # update age of received data
        [car.update_ages() for car in self._cars]
        
        # simulate transmission of delayed data among cars
        self.transmit_data(neigh_sizes)
        
        # simulate data reception at cars
        self.receive_data(data)
        
        # remove neighbors with age older than maximum allowed
        [car.remove_old_data() for car in self._cars]
        
        # store current edge attributes with received delayed data
        self._data = self.add_edge_attributes(data)

        # the episode ends when reaching max_episode_steps or all the agents reach the goal
        time_up = self._t >= self.max_episode_steps
        reach = torch.less(torch.norm(self.data.states[:, :2] - self._goal[:, :2], dim=1), self._params['dist2goal']).all()
        done = time_up or reach

        # reward function
        reward = - (self.collision_mask(data).sum() * 100 / self.num_agents) - 1.0 + reach * 1000.

        # assess safety
        safe = float(self.collision_mask(data).sum() == 0)
        
        return self.data, float(reward), done, {'safe': safe}


    def transmit_data(self, neigh_sizes):        
        for car_idx, car in enumerate(self._cars):
            avg_del = np.ceil(neigh_sizes[car_idx] * self._params['poisson_coeff'])
            delay_tx = poisson.rvs(avg_del)
            car.store_delay(delay_tx)

    
    def receive_data(self, data):
        data.available_states = -torch.ones(self.num_agents, self.state_dim)
        for idx_car, car in enumerate(self._cars):
            neighbors_list, delay_list = car.data_delivered()
            if delay_list:
                data.available_states[idx_car] = self._states[-min(delay_list) - 1][idx_car]
            
            for idx_neighbors, neighbors in enumerate(neighbors_list):
                delay_rec = delay_list[idx_neighbors]
                for neighbor in neighbors:
                    state_diff = self._states[-delay_rec - 1][idx_car] - self._states[-delay_rec - 1][neighbor]
                    action_diff = self._actions[-delay_rec - 1][idx_car] - self._actions[-delay_rec - 1][neighbor]
                    state_diff.to(self.device)
                    action_diff.to(self.device)
                    action_state_diff = torch.cat([action_diff, state_diff])
                    self._cars[neighbor].store_data(
                        idx_car, action_state_diff, delay_rec, self.delay_aware)
          
    
    def forward_graph(self, data: Data, action: Tensor) -> Data:
        
        # calculate next state using dynamics
        action = action + self.u_ref(data)
        lower_lim, upper_lim = self.action_lim
        action = torch.clamp(action, lower_lim, upper_lim)
        state = self.forward(data.states, action)

        # construct the graph of the next step, retaining the connection
        data_next = Data(
            x=torch.zeros(state.shape[0], self.node_dim).type_as(state),
            pos=state[:, :2],
            states=state,
            edge_index=data.edge_index
        )
        data_next.state_diff = data_next.states[data_next.edge_index[0]] - data_next.states[data_next.edge_index[1]]

        # increase age of all data
        if isinstance(data.edge_attr, PackedSequence):
            edge_attr_tmp = data.edge_attr
        else:
            edge_attr_tmp = pack_sequence(data.edge_attr, enforce_sorted=False)
            
        age_step = torch.zeros_like(edge_attr_tmp.data)
        age_step[:,-1] = 1.
        edge_attr = edge_attr_tmp._replace(data=edge_attr_tmp.data + age_step)
        data_next.edge_attr = edge_attr

        return data_next

    
    def render(
            self, traj: Optional[Tuple[Data, ...]] = None, return_ax: bool = False, plot_edge: bool = True
    ) -> Union[Tuple[np.array, ...], np.array]:
        return_tuple = True
        if traj is None:
            data = self.data
            traj = (data,)
            return_tuple = False

        r = self._params['car_radius']
        gif = []
        for data in traj:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=80)

            # plot the cars and the communication network
            plot_graph(ax, data, radius=r, color='#FF8C00', with_label=True,
                       plot_edge=plot_edge, alpha=0.8)
            pos_np = data.pos.cpu().detach().numpy()
            states_np = data.states.cpu().detach().numpy()
            for i in range(data.pos.shape[0]):
                ax.arrow(pos_np[i, 0] - r * np.cos(states_np[i, 2]), pos_np[i, 1] - r * np.sin(states_np[i, 2]),
                         2 * r * np.cos(states_np[i, 2]), 2 * r * np.sin(states_np[i, 2]))

            # plot the goals
            goal_data = Data(pos=self._goal[:, :2])
            plot_graph(ax, goal_data, radius=r, color='#3CB371',
                       with_label=True, plot_edge=False, alpha=0.8)
            
            # texts
            fontsize = 14
            collision_text = ax.text(0., 0.97, "", transform=ax.transAxes, fontsize=fontsize)
            unsafe_index = mask_to_index(self.collision_mask(data))
            collision_text.set_text(f'Collision: {unsafe_index.cpu().detach().numpy()}')

            # set axis limit
            x_interval = self._xy_max[0] - self._xy_min[0]
            y_interval = self._xy_max[1] - self._xy_min[1]
            ax.set_xlim(self._xy_min[0], self._xy_min[0] + max(x_interval, y_interval))
            ax.set_ylim(self._xy_min[1], self._xy_min[1] + max(x_interval, y_interval))
            plt.axis('off')

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
        

    def add_communication_links(self, data: Data) -> list:
        data = self._builder(data)
        neigh_sizes = [0] * self.num_agents
        for car_idx, car in enumerate(self._cars):
            neighbors = data.edge_index[0][data.edge_index[1] == car_idx].tolist()
            car.store_neighbors(neighbors)
            neigh_sizes[car_idx] = len(neighbors)
            
        return neigh_sizes
    

    def add_edge_attributes(self, data: Data) -> Data:
        edge_index = [[], []]
        edge_attr = []
        for car_idx, car in enumerate(self._cars):
            for neighbor in car.neighbor_data:
                edge_index[1].append(car_idx)
                edge_index[0].append(neighbor)
                edge_attr.append(car.neighbor_data[neighbor])
                
        data.edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        data.edge_attr = edge_attr
        data.state_diff = data.states[data.edge_index[0]] - data.states[data.edge_index[1]]
        
        return data
    

    def u_ref(self, data: Data) -> Tensor:
        states = data.states.reshape(-1, self.num_agents, self.state_dim)
        diff = (states - self._goal).reshape(-1, self.state_dim)
        states = states.reshape(-1, self.state_dim)

        # PID parameters
        k_omega = 0.2
        k_v = 0.3
        k_a = 0.6

        dist = torch.norm(diff[:, :2], dim=-1)
        theta_t = (torch.acos(-diff[:, 0] / (dist + 0.0001)) * torch.sign(-diff[:, 1])) % (2 * torch.pi)
        theta = states[:, 2] % (2 * torch.pi)
        theta_diff = theta_t - theta
        omega = torch.zeros(states.shape[0]).type_as(states)
        agent_dir = torch.cat([torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)], dim=-1)
        theta_between = torch.acos(torch.clamp(torch.bmm(-diff[:, :2].unsqueeze(1), agent_dir.unsqueeze(-1)).squeeze() / (dist + 0.0001), -1, 1))

        # when theta <= pi
        small_anti_clock_id = torch.where(torch.logical_and(torch.logical_and(theta_diff < torch.pi, theta_diff >= 0), theta <= torch.pi))
        small_clock_id = torch.where(torch.logical_and(torch.logical_not(torch.logical_and(theta_diff < torch.pi, theta_diff >= 0)), theta <= torch.pi))
        omega[small_anti_clock_id] = k_omega * theta_between[small_anti_clock_id]
        omega[small_clock_id] = -k_omega * theta_between[small_clock_id]

        # when theta > pi
        large_clock_id = torch.where(torch.logical_and(torch.logical_and(theta_diff > -torch.pi, theta_diff <= 0), theta > torch.pi))
        large_anti_clock_id = torch.where(torch.logical_and(torch.logical_not(torch.logical_and(theta_diff > -torch.pi, theta_diff <= 0)), theta > torch.pi))
        omega[large_clock_id] = -k_omega * theta_between[large_clock_id]
        omega[large_anti_clock_id] = k_omega * theta_between[large_anti_clock_id]
        omega = omega / (dist + 0.0001)
        omega = torch.clamp(omega, -5., 5.)

        a = -k_a * states[:, 3] + k_v * torch.norm(diff[:, :2], dim=-1)
        action = torch.cat([omega.unsqueeze(-1), a.unsqueeze(-1)], dim=-1)

        return action.reshape(-1, self.action_dim)

    def safe_mask(self, data: Data) -> Tensor:
        mask = torch.empty(data.states.shape[0]).type_as(data.x)
        for i in range(data.states.shape[0] // self.num_agents):
            state_diff = data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(1) - \
                         data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(0)
            pos_diff = state_diff[:, :, :2]
            dist = pos_diff.norm(dim=2)
            dist += torch.eye(dist.shape[0], device=self.device) * (2 * self._params['car_radius'] + 1)
            safe = torch.greater(dist, 4 * self._params['car_radius'])
            mask[self.num_agents * i: self.num_agents * (i + 1)] = torch.min(safe, dim=1)[0]

        mask = mask.bool()

        return mask


    def unsafe_mask(self, data: Data) -> Tensor:
        mask = torch.empty(data.states.shape[0]).type_as(data.x)
        warn_dist = 4 * self._params['car_radius']

        for i in range(data.states.shape[0] // self.num_agents):
            
            # collision
            state_diff = data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(1) - \
                         data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(0)
            pos_diff = state_diff[:, :, :2]
            dist = pos_diff.norm(dim=2)
            dist += torch.eye(dist.shape[0], device=self.device) * (2 * self._params['car_radius'] + 1)
            collision = torch.less(dist, 2 * self._params['car_radius'])
            mask[self.num_agents * i: self.num_agents * (i + 1)] = torch.max(collision, dim=1)[0]

            # dangerous direction
            warn_zone = torch.less(dist, warn_dist)
            pos_vec = pos_diff / (torch.norm(pos_diff, dim=2, keepdim=True) + 0.0001)  # [i, j]: j -> i
            theta_state = data.states[self.num_agents * i: self.num_agents * (i + 1), 2].reshape(self.num_agents, 1)
            theta_vec = torch.cat([torch.cos(theta_state), torch.sin(theta_state)], dim=1).repeat(self.num_agents, 1, 1)  # [i, j]: theta[j]
            inner_prod = torch.sum(pos_vec * theta_vec, dim=2)
            unsafe_threshold = torch.cos(torch.asin(self._params['car_radius'] * 2 / (dist + 0.0000001)))
            unsafe = torch.greater(inner_prod, unsafe_threshold)
            unsafe = torch.max(torch.logical_and(unsafe, warn_zone), dim=1)[0]

            mask[self.num_agents * i: self.num_agents * (i + 1)] = \
                torch.logical_or(mask[self.num_agents * i: self.num_agents * (i + 1)], unsafe)

        return mask.bool()

    def collision_mask(self, data: Data) -> Tensor:
        mask = torch.empty(data.states.shape[0]).type_as(data.x)
        for i in range(data.states.shape[0] // self.num_agents):
            state_diff = data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(1) - \
                         data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(0)
            pos_diff = state_diff[:, :, :2]
            dist = pos_diff.norm(dim=2)
            dist += torch.eye(dist.shape[0], device=self.device) * (2 * self._params['car_radius'] + 1)
            collision = torch.less(dist, 2 * self._params['car_radius'])
            mask[self.num_agents * i: self.num_agents * (i + 1)] = torch.max(collision, dim=1)[0]
        
        mask = mask.bool()

        return mask
