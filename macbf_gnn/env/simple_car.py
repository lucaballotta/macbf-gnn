import numpy as np
import torch
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import Tuple, Optional, Union
from collections import deque
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms.radius_graph import RadiusGraph
from torch_geometric.utils import mask_to_index
from torch.nn.utils.rnn import pack_sequence
from cvxpy import Expression
from scipy.stats import poisson
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, PackedSequence

from .utils import lqr, plot_graph
from .base import Agent, MultiAgentEnv


class SimpleCars(MultiAgentEnv):

    def __init__(self, num_agents: int, device: torch.device, dt: float = 0.05, params: dict = None, delay_aware: bool = True):
        super(SimpleCars, self).__init__(num_agents, device, dt, params, delay_aware)
        
        # state trajectory
        self._states = deque(maxlen=self._params['buffer_size'])
        
        # cars
        car = SimpleCar(buffer_size=self._params['buffer_size'], max_age=self._params['max_age'])
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
            # 'comm_radius': 1.0,  # communication radius
            # 'car_radius': 0.05,  # radius of the cars
            # 'dist2goal': 0.05,  # goal reaching threshold
            'speed_limit': 0.4,  # maximum speed
            'max_distance': 1.5,  # maximum moving distance to goal
            'area_size': 3.0,
            'car_radius': 0.05,
            'dist2goal': 0.02,
            'comm_radius': 1.0,
            'buffer_size': 1,
            'max_age': 2,
            'poisson_coeff': .1,
            'test_epi_max_iters': 200
        }
        
        
    def dynamics(self, x: Tensor, u: Union[Tensor, Expression]) -> Union[Tensor, Expression]:
        if isinstance(u, Expression):
            x = x.cpu().detach().numpy()
            A = np.zeros((self.state_dim, self.state_dim))
            A[0, 2] = 1.
            A[1, 3] = 1.
            B = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
            xdot = x @ A.T + u @ B.T
            return xdot
        else:
            return torch.cat([x[:, 2:], u], dim=1)
        
        
    def reset(self) -> Data:
        self._t = 0
        # side_length = np.sqrt(max(1.0, self.num_agents / 8.0))
        # side_length = np.sqrt(self._params['car_radius'] * 10 * self.num_agents)
        # side_length = np.sqrt(self._params['car_radius'] * (10 + 2) * 8)
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
                # candidate = torch.rand(2, device=self.device) * side_length
                candidate = (torch.rand(2, device=self.device) * 2 - 1) * self._params['max_distance'] + states[i]
                dist_min = torch.norm(goals - candidate, dim=1).min()
                if dist_min <= self._params['car_radius'] * 4:
                    continue
                goals[i] = candidate
                i += 1
        elif self._mode == 'demo_0':
            for i in range(self.num_agents // 2):
                states[i] = torch.tensor([i * self._params['car_radius'] * 10, 0], device=self.device)
                states[i + self.num_agents // 2] = torch.tensor(
                    [i * self._params['car_radius'] * 10 + self._params['car_radius'],
                     self._params['car_radius'] * 20], device=self.device)
                goals[i] = states[i + self.num_agents // 2] - self._params['car_radius']
                goals[i + self.num_agents // 2] = states[i] + self._params['car_radius']
        else:
            raise ValueError('Reset environment: unknown type of mode!')
            
        # record goals
        self._goal = goals

        # add velocity
        states = torch.cat([states, torch.zeros(self.num_agents, 2, device=self.device)], dim=1)

        # store states
        self._states.append(states)
        
        # reset agents' data
        [car.reset_data() for car in self._cars]

        # build graph
        data = Data(x=torch.zeros_like(states), pos=states[:, :2], states=states)
        neigh_sizes = self.add_communication_links(data)
        
        # simulate transmission of delayed data among cars
        self.transmit_data(neigh_sizes)
        
        # simulate data reception at cars
        self.receive_data()
        
        # remove neighbors with age older than maximum allowed
        [car.remove_old_data() for car in self._cars]
        
        # store current edge attributes with received delayed data
        self._data = self.add_edge_attributes(data)

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
        # print('')
        # print('action', action)
        # print('ref', self.u_ref(self._data))
        action = action + self.u_ref(self._data)
        lower_lim, upper_lim = self.action_lim
        action = torch.clamp(action, lower_lim, upper_lim)
        with torch.no_grad():
            state = self.forward(self.state, action)
            self._states.append(state)
        
        # construct graph using the new states
        data = Data(x=torch.zeros_like(state), pos=state[:, :2], states=state)
        neigh_sizes = self.add_communication_links(data)
        
        # update age of received data
        [car.update_ages() for car in self._cars]
        
        # simulate transmission of delayed data among cars
        self.transmit_data(neigh_sizes)
        
        # simulate data reception at cars
        self.receive_data()
        
        # remove neighbors with age older than maximum allowed
        [car.remove_old_data() for car in self._cars]
        
        # store current edge attributes with received delayed data
        self._data = self.add_edge_attributes(data)
        
        # the episode ends when reaching max_episode_steps or all the agents reach the goal
        time_up = self._t >= self.max_episode_steps
        reach = torch.less(torch.norm(self.data.states[:, :2] - self._goal, dim=1), self._params['dist2goal']).all()
        done = time_up or reach

        # reward function
        reward = - (self.unsafe_mask(data).sum() * 100 / self.num_agents) - 1.0 + reach * 1000.

        safe = self.unsafe_mask(data).sum() == 0
        return self.data, float(reward), done, {'safe': safe}
    
    
    def transmit_data(self, neigh_sizes):        
        for car_idx, car in enumerate(self._cars):
            avg_del = np.ceil(neigh_sizes[car_idx] * self._params['poisson_coeff'])
            delay_tx = 0 #poisson.rvs(avg_del)
            car.store_delay(delay_tx)
            

    def receive_data(self):
        for idx_car, car in enumerate(self._cars):
            neighbors_list, delay_list = car.data_delivered()
            for idx_neighbors, neighbors in enumerate(neighbors_list):
                delay_rec = delay_list[idx_neighbors]
                for neighbor in neighbors:
                    state_diff = self._states[-delay_rec - 1][idx_car] - self._states[-delay_rec - 1][neighbor]
                    stored, idx = self._cars[neighbor].store_data(
                        idx_car, state_diff, delay_rec, self.delay_aware)
                    # if stored and self.delay_aware:
                    #     self._cars[neighbor].store_states(
                    #         self._states[-delay_rec - 1][neighbor], idx_car, self._states[-delay_rec - 1][idx_car], idx)
                    

    def forward_graph(self, data: Data, action: Tensor) -> Data:
        
        # calculate next state using dynamics
        action = action + self.u_ref(data)
        lower_lim, upper_lim = self.action_lim
        action = torch.clamp(action, lower_lim, upper_lim)
        state = self.forward(data.states, action)
        
        # construct the graph of the next step, retaining the connection
        data_next = Data(
            x=torch.zeros_like(state),
            pos=state[:, :2],
            states=state,
            edge_index=data.edge_index,
            state_diff=state[data.edge_index[0]] - state[data.edge_index[1]]
        )
                
        # increase age of all data
        if isinstance(data.edge_attr, PackedSequence):
            edge_attr_tmp = data.edge_attr
        else:
            edge_attr_tmp = pack_sequence(data.edge_attr, enforce_sorted=False)
            
        age_step = torch.zeros_like(edge_attr_tmp.data)
        age_step[:,-1] = 1
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
                       plot_edge=plot_edge, alpha=0.8, danger_radius=2*r, safe_radius=3*r)

            # plot the goals
            goal_data = Data(pos=self._goal[:, :2])
            plot_graph(ax, goal_data, radius=r, color='#3CB371',
                       with_label=True, plot_edge=False, alpha=0.8)

            # texts
            fontsize = 14
            collision_text = ax.text(0., 0.97, "", transform=ax.transAxes, fontsize=fontsize)
            unsafe_index = mask_to_index(self.unsafe_mask(data))
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
                
        data.edge_index = torch.tensor(edge_index, dtype=torch.long)
        data.edge_attr = edge_attr
        data.state_diff = data.states[data.edge_index[0]] - data.states[data.edge_index[1]]
        
        # if len(edge_attr) > 0:
        #     if self.delay_aware:
        #         data.edge_attr = pack_sequence(edge_attr, enforce_sorted=False)
        #     else:
        #         data.edge_attr = torch.cat(edge_attr)
        # else:
        #     data.edge_attr = edge_attr
        
        return data


    @property
    def state_lim(self) -> Tuple[Tensor, Tensor]:
        low_lim = torch.tensor(
            [self._xy_min[0], self._xy_min[1], -self._params['speed_limit'], -self._params['speed_limit']],
            device=self.device)
        high_lim = torch.tensor(
            [self._xy_max[0], self._xy_max[1], self._params['speed_limit'], self._params['speed_limit']],
            device=self.device)
        return low_lim, high_lim
    
    
    @property
    def action_lim(self) -> Tuple[Tensor, Tensor]:
        upper_limit = torch.ones(2, device=self.device) * 10.
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
        mask = torch.empty(data.states.shape[0]).type_as(data.x)
        for i in range(data.states.shape[0] // self.num_agents):
            state_diff = data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(1) - \
                         data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(0)
            pos_diff = state_diff[:, :, :2]
            dist = pos_diff.norm(dim=2)
            dist += torch.eye(dist.shape[0], device=self.device) * (2 * self._params['car_radius'] + 1)
            safe = torch.greater(dist, 4 * self._params['car_radius'])  # 3 is a hyperparameter
            mask[self.num_agents * i: self.num_agents * (i + 1)] = torch.min(safe, dim=1)[0]
        mask = mask.bool()

        return mask
    
    
    def unsafe_mask(self, data: Data) -> Tensor:
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

class SimpleCar(Agent):
    
    def __init__(self, buffer_size, max_age):
        super().__init__(buffer_size, max_age)
        
    def store_data(self, neighbor: int, data: Tensor, delay: int, delay_aware: bool) -> Tuple[bool, int]:
        if delay_aware:
            stored = False
            idx = None
            if delay <= self.max_age:
                stored = True
                data = torch.reshape(data, (-1,))
                data = torch.unsqueeze(torch.cat((data, torch.tensor([delay]))), 0)
                if neighbor not in self._neighbor_data:
                    self._neighbor_data[neighbor] = data
                    
                else:
                    
                    # insertionSort: oldest data first
                    idx = 0
                    while self._neighbor_data[neighbor][idx][-1] > delay:
                        idx += 1
                        if idx == len(self._neighbor_data[neighbor]):
                            break

                    previous_data = self._neighbor_data[neighbor]
                    self._neighbor_data[neighbor] = torch.cat(
                        (previous_data[:idx], data, previous_data[idx:]))
                    
        else:
            stored = True
            idx = 0
            self._neighbor_data[neighbor] = data
        
        return stored, idx
    
    def store_states(self, self_state: Tensor, neighbor: int, neighbor_state: Tensor, idx: int):
        self_state = torch.reshape(self_state, (-1,))
        neighbor_state = torch.reshape(neighbor_state, (-1,))
        new_data = torch.unsqueeze(torch.stack((self_state, neighbor_state)), 0)
        if neighbor not in self._past_states:
            self._past_states[neighbor] = new_data
            
        else:
            previous_past_states = self._past_states[neighbor]
            self._past_states[neighbor] = torch.cat(
                (previous_past_states[:idx], new_data, previous_past_states[idx:]))    
            
    def update_ages(self):
        for neighbor in self.neighbor_data:
            for state_info in self.neighbor_data[neighbor]:
                state_info[-1] += 1
                
    def remove_old_data(self):
        old_neighbors = []
        for neighbor in self._neighbor_data:
            if self._neighbor_data[neighbor][-1][-1] > self.max_age:
                old_neighbors.append(neighbor)
                
        for neighbor in old_neighbors:
            del self._neighbor_data[neighbor]
            
        for neighbor in self._neighbor_data:
            idx = 0 
            while self._neighbor_data[neighbor][idx][-1] > self.max_age:
                idx += 1
                
            self._neighbor_data[neighbor] = self._neighbor_data[neighbor][idx:]
    
    def data_delivered(self):
        neighbors = []
        delays = []
        for idx, delay in enumerate(self._delays):
            if delay == len(self._delays) - 1 - idx:
                neighbors.append(self._neighbors[idx])
                delays.append(delay)
                
        return neighbors, delays