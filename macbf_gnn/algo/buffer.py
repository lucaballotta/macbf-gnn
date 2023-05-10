import numpy as np
import random

from typing import Tuple, Any, Union, Optional, List
from torch_geometric.data import Data


class Buffer:
    def __init__(self):
        self._data = []  # list with all graphs
        self._safe_data = []  # list of positions with safe graphs
        self._unsafe_data = []  # list of positions with unsafe graphs
        self.MAX_SIZE = 100000

    def append(self, data: Data, is_safe: bool):
        self._data.append(data)
        self._safe_data.append(self.size - 1) if is_safe else self._unsafe_data.append(self.size - 1)
        if self.size > self.MAX_SIZE:
            del self._data[0]  # remove oldest data
            try:
                self._safe_data.remove(0)
            except ValueError:
                self._unsafe_data.remove(0)

            # todo: this can be optimized by using fixed-length buffer and record the current position
            self._safe_data = [i - 1 for i in self._safe_data]
            self._unsafe_data = [i - 1 for i in self._unsafe_data]

    @property
    def data(self) -> List[Data]:
        return self._data
    
    @property
    def safe_data(self) -> List[int]:
        return self._safe_data
    
    @property
    def unsafe_data(self) -> List[int]:
        return self._unsafe_data

    @property
    def size(self) -> int:
        return len(self._data)

    def merge(self, other):
        size_init = self.size
        self._data += other.data
        other_safe_data = [i + size_init for i in other.safe_data]
        self._safe_data.extend(other_safe_data)
        other_unsafe_data = [i + size_init for i in other.unsafe_data]
        self._unsafe_data.extend(other_unsafe_data)
        if self.size > self.MAX_SIZE:
            to_delete = self.size - self.MAX_SIZE
            for i in range(to_delete):
                try:
                    self._safe_data.remove(i)
                    
                except:
                    self._unsafe_data.remove(i)
                    
            self._safe_data = [i - to_delete for i in self._safe_data]
            self._unsafe_data = [i - to_delete for i in self._unsafe_data]
            del self._data[:to_delete] # remove oldest data

    def clear(self):
        self._data.clear()
        self._safe_data.clear()
        self._unsafe_data.clear()

    def sample(self, n: int, balanced_sampling: bool=False) -> List[Data]:
        """
        Sample at random states of trajectory from buffer.

        Parameters
        ----------
        n: int,
            number of samples
        balanced_sampling: bool,
            balance the samples from safe states and unsafe states
        """
        data_list = []
        if not balanced_sampling:
            index = np.sort(np.random.randint(0, self.size, n))

        else:
            index_unsafe, index_safe = [], []
            if len(self.unsafe_data) > 0:
                if len(self.unsafe_data) > n // 2:
                    index_unsafe = random.choices(self.unsafe_data, k=n // 2)
                else:
                    index_unsafe = self.unsafe_data
                    
            if len(self.safe_data) > 0:
                if len(self.safe_data) > n // 2:
                    index_safe = random.choices(self.safe_data, k=n // 2)
                else:
                    index_safe = self.safe_data
                    
            index = sorted(index_safe + index_unsafe)

        for i in index:
            data_list.append(self._data[i])

        return data_list