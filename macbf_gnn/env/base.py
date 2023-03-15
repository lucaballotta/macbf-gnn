import numpy as np
import torch
import copy

from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple, Optional, Union, List
from torch import Tensor
from torch_geometric.data import Data
from cvxpy import Variable, Expression
from collections import deque


class MultiAgentEnv(ABC):

    def __init__(self, num_agents: int, device: torch.device, dt: float = 0.05, params: dict = None):
        super(MultiAgentEnv, self).__init__()
        self._num_agents = num_agents
        self._device = device
        self._dt = dt
        if params is None:
            params = self.default_params
        self._params = params
        self._data = None
        self._t = 0
        self._mode = 'train'

    def train(self):
        self._mode = 'train'

    def test(self):
        self._mode = 'test'

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def dt(self) -> float:
        """
        Returns
        -------
        dt: float,
            simulation time interval
        """
        return self._dt

    @property
    def device(self) -> torch.device:
        """
        Returns
        -------
        device: torch.device
            device of all the Tensors
        """
        return self._device

    @property
    def data(self) -> Data:
        """
        Returns
        -------
        data: Data,
            data of the current graph
        """
        return self._data

    @property
    def state(self) -> Tensor:
        """
        Returns
        -------
        state: Tensor (n, state_dim),
            the agents' states
        """
        return self._data.states

    @abstractproperty
    def state_dim(self) -> int:
        """
        Returns
        -------
        state_dim: int,
            dimension of the state
        """
        pass

    @abstractproperty
    def node_dim(self) -> int:
        """
        Returns
        -------
        node_dim: int,
            dimension of the node information
        """
        pass

    @abstractproperty
    def edge_dim(self) -> int:
        """
        Returns
        -------
        edge_dim: int,
            dimension of the edge information
        """
        pass

    @abstractproperty
    def action_dim(self) -> int:
        """
        Returns
        -------
        action_dim: int,
            dimension of the control action
        """
        pass

    @abstractproperty
    def max_episode_steps(self) -> int:
        """
        Get maximum simulation time steps.
        The simulation will be ended if the time step exceeds the maximum time steps.

        Returns
        -------
        max_episode_steps: int,
            maximum simulation time steps
        """
        pass

    @abstractproperty
    def default_params(self) -> dict:
        """
        Get default parameters.

        Returns
        -------
        params: dict,
            a dict of default parameters
        """
        pass

    @abstractmethod
    def dynamics(self, x: Tensor, u: Union[Tensor, Expression]) -> Union[Tensor, Expression]:
        """
        Dynamics of a single agent \dot{x} = f(x, u).

        Parameters
        ----------
        x: Tensor (bs, state_dim),
            current state of the agent
        u: Union[Tensor, Expression] (bs, action_dim) or (action_dim,),
            control input

        Returns
        -------
        xdot: Union[Tensor, Expression] (bs, state_dim),
            time derivative of the state
        """
        pass

    @abstractmethod
    def reset(self) -> Data:
        """
        Reset the environment and return the current graph.

        Returns
        -------
        data: Data,
            data of the current graph
        """
        pass

    @abstractmethod
    def step(self, action: Tensor) -> Tuple[Data, float, bool, dict]:
        """
        Simulation the system for one step.

        Parameters
        ----------
        action: Tensor (n, action_dim),
            action of all the agents

        Returns
        -------
        next_data: Data,
            graph data of the next time step
        reward: float,
            reward signal
        done: bool,
            if the simulation is ended or not
        info: dict,
            other useful information, including safe or unsafe
        """
        pass

    @abstractmethod
    def forward_graph(self, data: Data, action: Tensor) -> Data:
        """
        Get the graph of the next timestep after doing the action.
        The connection of the graph will be retained.

        Parameters
        ----------
        data: Data,
            batched graph data using Batch.from_datalist
        action: Tensor (bs x n, action_dim),
            action of all the agents in the batch

        Returns
        -------
        next_data: Data,
            batched graph data of the next time step
        """
        pass

    @torch.no_grad()
    def edge_dynamics(self, data: Data, action: Variable) -> Variable:
        action = action + self.u_ref(data).cpu().numpy()
        state_dot = self.dynamics(data.states, action)
        edge_index = data.edge_index.cpu().numpy()
        edge_dot = state_dot[edge_index[0]] - state_dot[edge_index[1]]
        return edge_dot

    @abstractmethod
    def render(
            self, traj: Optional[Tuple[Data, ...]] = None, return_ax: bool = False, plot_edge: bool = True
    ) -> Union[Tuple[np.array, ...], np.array]:
        """
        Plot the environment for the current time step.
        If traj is not None, plot the environment for the trajectory.

        Parameters
        ----------
        traj: Optional[Tuple[Data, ...]],
            a tuple of Tensor containing the graph of the trajectories.

        Returns
        -------
        fig: numpy array,
            if traj is None: an array of the figure of the current environment
            if traj is not None: an array of the figures of the trajectory
        """
        pass

    @abstractmethod
    def add_communication_links(self, data: Data) -> List[int]:
        """
        Add communication links to the graph at the current time and stores links into agents' data.

        Parameters
        ----------
        data: Data,
            graph data at current time

        Returns
        -------
        neigh_sizes: List[int],
            dimension of neighborhoods of agents at current time
        """
        pass
    
    @abstractmethod
    def add_edge_attributes(self, data: Data) -> Data:
        """
        Add communication links to the graph at the current time and stores links into agents' data.

        Parameters
        ----------
        data: Data,
            graph data at current time

        Returns
        -------
        data: Data,
            graph data with edge_index and edge_attr of delayed data received at current time
        """
        pass

    @abstractmethod
    def transmit_data(self, neigh_sizes: List[int]):
        """
        Simulate data transmission and reception with communication delays.
        """
        pass

    @abstractproperty
    def state_lim(self) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        lower limit, upper limit: Tuple[Tensor, Tensor],
            limits of the state
        """
        pass

    @abstractproperty
    def action_lim(self) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[Tensor, Tensor],
            limits of the action
        """
        pass

    @abstractmethod
    def u_ref(self, data: Data) -> Tensor:
        """
        Get reference control to finish the task without considering safety.

        Parameters
        ----------
        data: Data,
            current graph

        Returns
        -------
        u_ref: Tensor (bs x n, action_dim)
            reference control signal
        """
        pass

    @abstractmethod
    def safe_mask(self, data: Data) -> Tensor:
        """
        Mask out the safe agents. Masks are applied to each agent indicating safe (1) or dangerous (0)

        Parameters
        ----------
        data: Data,
            current network

        Returns
        -------
        mask: Tensor (bs x n, mask),
            the agent is safe (1) or unsafe (0)
        """
        pass

    @abstractmethod
    def unsafe_mask(self, data: Data) -> Tensor:
        """
        Mask out the unsafe agents. Masks are applied to each agent indicating safe (0) or dangerous (1)

        Parameters
        ----------
        data: Data,
            current network

        Returns
        -------
        mask: Tensor (bs x n, mask),
            the agent is safe (0) or unsafe (1)
        """
        pass

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Simulate the single agent for one time step.

        Parameters
        ----------
        x: Tensor (bs x n, state_dim),
            current state of the agent
        u: Tensor (bs x n, action_dim),
            control input

        Returns
        -------
        x_next: Tensor (bs x n, state_dim),
            next state of the agent
        """
        xdot = self.dynamics(x, u)
        return x + xdot * self.dt


class Agent(ABC):
    
    def __init__(self, buffer_size, max_age):
        super(Agent, self).__init__()
        self.buffer_size = buffer_size
        self.max_age = max_age
        self._neighbors = deque(maxlen = self.buffer_size)
        self._delays = deque(maxlen = self.buffer_size)
        self._neighbor_data = dict()
        
    @property
    def neighbor_data(self):
        return self._neighbor_data
    
    def reset_data(self):
        self._neighbors.clear()
        self._delays.clear()
        self._neighbor_data.clear()
    
    def copy(self):
        return copy.deepcopy(self)
    
    def store_delay(self, delay: int):
        """
        Store communication delay for transmission started at current time.

        Parameters
        ----------
        delay: int,
            communication delay
        """
        self._delays.append(delay)
        
    def store_neighbors(self, neighbors: List[int]):
        """
        Store agents to which data are transmitted at current time.

        Parameters
        ----------
        neighbors: List[int],
            indices of neighbors
        """
        self._neighbors.append(neighbors)
        
    @abstractmethod
    def store_data(self, neighbor: int, data: Tensor, delay: int) -> bool:
        """
        Store data that have been received from neighbors at the current time step.

        Parameters
        ----------
        neighbor: int,
            index of the neighbor
        data: Tensor (state_dim),
            state information received from the neighbor
        delay: int,
            communication delay after which data are received
            
        Returns
        -------
        data_stored: bool,
            True if data can be stored, False otherwise
        """
        pass
            
    @abstractmethod
    def update_ages(self):
        """
        Update AoI of stored data.
        """
        pass
                
    @abstractmethod
    def remove_old_data(self):
        """
        Remove stored data whose AoI is too large.
        """
        pass