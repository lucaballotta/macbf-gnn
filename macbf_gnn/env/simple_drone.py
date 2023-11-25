import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import Tuple, Optional, Union
from collections import deque
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms.radius_graph import RadiusGraph
from torch_geometric.utils import mask_to_index
from torch.nn.utils.rnn import pack_sequence
from cvxpy import Expression
from scipy.stats import poisson
from torch.nn.utils.rnn import pack_sequence, PackedSequence

from .utils import lqr, plot_graph_3d
from .base import Agent, MultiAgentEnv


class SimpleDrone(MultiAgentEnv):

    def __init__(self, num_agents: int, device: torch.device, dt: float = 0.05, params: dict = None, delay_aware: bool = True):
        super(SimpleDrone, self).__init__(num_agents, device, dt, params, delay_aware)
        
        # states and actions along trajectory
        self._states = deque(maxlen=self._params['buffer_size'])
        self._actions = deque(maxlen=self._params['buffer_size'])  # actions that generate the states in self._states from previous states
        
        # cars
        drone = Agent(buffer_size=self._params['buffer_size'], max_age=self._params['max_age'], device=self.device)
        self._drones = [drone.copy() for _ in range(self.num_agents)]

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
        return 3

    @property
    def node_dim(self) -> int:
        return 3

    @property
    def edge_dim(self) -> int:
        return 3

    @property
    def action_dim(self) -> int:
        return 3
    
    @property
    def state_lim(self) -> Tuple[Tensor, Tensor]:
        low_lim = torch.tensor(
            [self._xyz_min[0], self._xyz_min[1], self._xyz_min[2]],
            device=self.device)
        high_lim = torch.tensor(
            [self._xyz_max[0], self._xyz_max[1], self._xyz_max[2]],
            device=self.device)
        return low_lim, high_lim
    
    @property
    def action_lim(self) -> Tuple[Tensor, Tensor]:
        upper_limit = self._params['speed_limit']
        lower_limit = - upper_limit
        return lower_limit, upper_limit
    
    @property
    def max_episode_steps(self) -> int:
        return 500

    @property
    def default_params(self) -> dict:
        return {
            'speed_limit': 0.4,  # maximum speed
            'max_distance': 1.5,  # maximum moving distance to goal
            'area_size': 2.0,
            'drone_radius': 0.05,  # radius of the car
            'dist2goal': 0.02,  # goal reaching threshold
            'comm_radius': 0.5,  # communication radius
            'buffer_size': 5,  # max number of transmissions (tx delays) stored by cars
            'max_age': 5,  # max age of data stored by cars (older are discarded)
            'poisson_coeff': .5
        }
        
        
    def dynamics(self, x: Tensor, u: Union[Tensor, Expression]) -> Union[Tensor, Expression]:
        return u
        
        
    def reset(self) -> Data:
        self._t = 0
        side_length = self._params['area_size']
        states = torch.zeros(self.num_agents, 3, device=self.device)
        goals = torch.zeros(self.num_agents, 3, device=self.device)
        if self._mode == 'train' or self._mode == 'test':
            
            # randomly generate positions of agents
            i = 0
            while i < self.num_agents:
                candidate = torch.rand(3, device=self.device) * side_length
                dist_min = torch.norm(states - candidate, dim=1).min()
                if dist_min <= self._params['drone_radius'] * 4:
                    continue
                states[i] = candidate
                i += 1

            # randomly generate goals of agents
            i = 0
            while i < self.num_agents:
                candidate = torch.rand(3, device=self.device) * side_length
                dist_min = torch.norm(goals - candidate, dim=1).min()
                if dist_min <= self._params['drone_radius'] * 4:
                    continue
                goals[i] = candidate
                i += 1

        elif self._mode == 'demo_0':
            for i in range(self.num_agents // 2):
                states[i] = torch.tensor([i * self._params['drone_radius'] * 10, 0], device=self.device)
                states[i + self.num_agents // 2] = torch.tensor(
                    [i * self._params['drone_radius'] * 10 + self._params['drone_radius'],
                     self._params['drone_radius'] * 20], device=self.device)
                goals[i] = states[i + self.num_agents // 2] - self._params['drone_radius']
                goals[i + self.num_agents // 2] = states[i] + self._params['drone_radius']

        else:
            raise ValueError('Reset environment: unknown type of mode!')
            
        # record goals
        self._goal = goals

        # store states
        self._states.append(states)
        
        # initialize actions to zero
        self._actions.append(torch.zeros(self.num_agents, self.action_dim, device=self.device))
        
        # reset agents' data
        [drone.reset_data() for drone in self._drones]

        # build graph
        data = Data(
            x=torch.zeros_like(states), 
            pos=states, 
            states=states)
        neigh_sizes = self.add_communication_links(data)
        
        # simulate transmission of delayed data among cars
        self.transmit_data(neigh_sizes)
        
        # simulate data reception at cars
        self.receive_data(data)
        
        if self.delay_aware:

            # remove neighbors with age older than maximum allowed
            [drone.remove_old_data() for drone in self._drones]
        
        # store current edge attributes with received delayed data
        self._data = self.add_edge_attributes(data)

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
            self._states.append(state)
        
        # construct graph using the new states
        data = Data(x=torch.zeros_like(state), pos=state, states=state)
        neigh_sizes = self.add_communication_links(data)
        
        # update age of received data
        [drone.update_ages() for drone in self._drones]
        
        # simulate transmission of delayed data among cars
        self.transmit_data(neigh_sizes)
        
        # simulate data reception at cars
        self.receive_data(data)
        
        # remove neighbors with age older than maximum allowed
        [drone.remove_old_data() for drone in self._drones]
        
        # store current edge attributes with received delayed data
        self._data = self.add_edge_attributes(data)
                
        # the episode ends when reaching max_episode_steps or all the agents reach the goal
        time_up = self._t >= self.max_episode_steps
        reach = torch.less(torch.norm(self.data.states - self._goal, dim=1), self._params['dist2goal']).all()
        done = time_up or reach

        # reward function
        reward = - (self.unsafe_mask(data).sum() * 100 / self.num_agents) - 1.0 + reach * 1000.

        # assess safety
        safe = self.unsafe_mask(data).sum() == 0

        return self.data, float(reward), done, {'safe': safe}
    
    
    def transmit_data(self, neigh_sizes):        
        for drone_idx, drone in enumerate(self._drones):
            avg_del = np.ceil(neigh_sizes[drone_idx] * self._params['poisson_coeff'])
            delay_tx = poisson.rvs(avg_del)
            drone.store_delay(delay_tx)
            

    def receive_data(self, data):
        data.available_states = -torch.ones(self.num_agents, self.state_dim)
        for idx_drone, drone in enumerate(self._drones):
            neighbors_list, delay_list = drone.data_delivered()
            if delay_list:
                data.available_states[idx_drone] = self._states[-min(delay_list) - 1][idx_drone]
            
            for idx_neighbors, neighbors in enumerate(neighbors_list):
                delay_rec = delay_list[idx_neighbors]
                for neighbor in neighbors:
                    state_diff = self._states[-delay_rec - 1][idx_drone] - self._states[-delay_rec - 1][neighbor]
                    state_diff.to(self.device)
                    if self._delay_aware:
                        action_diff = self._actions[-delay_rec - 1][idx_drone] - self._actions[-delay_rec - 1][neighbor]
                        action_diff.to(self.device)
                        stored_data = torch.cat([action_diff, state_diff])
                    
                    else:
                        stored_data = state_diff

                    self._drones[neighbor].store_data(
                        idx_drone, stored_data, delay_rec, self.delay_aware)
                    

    def forward_graph(self, data: Data, action: Tensor) -> Data:
        
        # calculate next state using dynamics
        action = action + self.u_ref(data)
        lower_lim, upper_lim = self.action_lim
        action = torch.clamp(action, lower_lim, upper_lim)
        state = self.forward(data.states, action)
        
        # construct the graph of the next step, retaining the connection
        data_next = Data(
            x=torch.zeros_like(state),
            pos=state,
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

        r = self._params['drone_radius']
        gif = []
        for data in traj:
            fig = plt.figure(figsize=(10, 10), dpi=80)
            ax = fig.add_subplot(projection='3d')

            # plot the drones and the communication network
            plot_graph_3d(ax, data, radius=r, color='#FF8C00', with_label=True,
                          plot_edge=plot_edge, alpha=0.3)
            
            # plot the goals
            goal_data = Data(pos=self._goal[:, :2])
            plot_graph_3d(ax, goal_data, radius=r, color='#3CB371', with_label=True,
                          plot_edge=False, alpha=0.3)

            # texts
            fontsize = 14
            collision_text = ax.text(0., 0.97, "", transform=ax.transAxes, fontsize=fontsize)
            unsafe_index = mask_to_index(self.unsafe_mask(data))
            collision_text.set_text(f'Collision: {unsafe_index.cpu().detach().numpy()}')

            # set axis limit
            ax.set_xlim(self._xyz_min[0], self._xyz_max[0])
            ax.set_ylim(self._xyz_min[1], self._xyz_max[1])
            ax.set_zlim(self._xyz_min[2], self._xyz_max[2])
            ax.set_aspect('equal')

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
        for drone_idx, drone in enumerate(self._drones):
            neighbors = data.edge_index[0][data.edge_index[1] == drone_idx].tolist()
            drone.store_neighbors(neighbors)
            neigh_sizes[drone_idx] = len(neighbors)
            
        return neigh_sizes
        
        
    def add_edge_attributes(self, data: Data) -> Data:
        edge_index = [[], []]
        edge_attr = []
        for drone_idx, drone in enumerate(self._drones):
            for neighbor in drone.neighbor_data:
                edge_index[1].append(drone_idx)
                edge_index[0].append(neighbor)
                edge_attr.append(drone.neighbor_data[neighbor])
                
        data.edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        data.edge_attr = edge_attr
        data.state_diff = data.states[data.edge_index[0]] - data.states[data.edge_index[1]]
        
        return data
    
    
    def u_ref(self, data: Data) -> Tensor:
        states = data.states.reshape(-1, self.num_agents, self.state_dim)
        diff = (states - self._goal)

        if self._K is None:
            # calculate the LQR controller
            A = np.eye(self.state_dim)
            B = np.eye(self.state_dim) * self.dt
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
            pos_diff = state_diff[:, :, :3]
            dist = pos_diff.norm(dim=2)
            dist += torch.eye(dist.shape[0], device=self.device) * (2 * self._params['drone_radius'] + 1)
            safe = torch.greater(dist, 4 * self._params['drone_radius'])
            mask[self.num_agents * i: self.num_agents * (i + 1)] = torch.min(safe, dim=1)[0]

        return mask.bool()
    
    
    def unsafe_mask(self, data: Data) -> Tensor:
        mask = torch.empty(data.states.shape[0]).type_as(data.x)
        for i in range(data.states.shape[0] // self.num_agents):
            state_diff = data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(1) - \
                data.states[self.num_agents * i: self.num_agents * (i + 1)].unsqueeze(0)
            pos_diff = state_diff[:, :, :3]
            dist = pos_diff.norm(dim=2)
            dist += torch.eye(dist.shape[0], device=self.device) * (2 * self._params['drone_radius'] + 1)
            collision = torch.less(dist, 2 * self._params['drone_radius'])
            mask[self.num_agents * i: self.num_agents * (i + 1)] = torch.max(collision, dim=1)[0]
        
        return mask.bool()
    