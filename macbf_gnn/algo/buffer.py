import numpy as np
import random

from typing import Tuple, Any, Union, Optional, List
from torch_geometric.data import Data


class Buffer:
    def __init__(self):
        self._data = []  # list with all graphs
        self.safe_data = []  # list of positions with safe graphs
        self.unsafe_data = []  # list of positions with unsafe graphs
        self.MAX_SIZE = 100000

    def append(self, data: Data, is_safe: bool):
        self._data.append(data)
        self.safe_data.append(self.size - 1) if is_safe else self.unsafe_data.append(self.size - 1)
        if self.size > self.MAX_SIZE:
            del self._data[0]  # remove oldest data
            try:
                self.safe_data.remove(0)
            except ValueError:
                self.unsafe_data.remove(0)

            # todo: this can be optimized by using fixed-length buffer and record the current position
            self.safe_data = [i - 1 for i in self.safe_data]
            self.unsafe_data = [i - 1 for i in self.unsafe_data]

    @property
    def data(self) -> List[Data]:
        return self._data

    @property
    def size(self) -> int:
        return len(self._data)

    def merge(self, other):
        size_init = self.size
        self._data += other.data
        other_safe_data = [i + size_init for i in other.safe_data]
        self.safe_data.extend(other_safe_data)
        other_unsafe_data = [i + size_init for i in other.unsafe_data]
        self.unsafe_data.extend(other_unsafe_data)
        if self.size > self.MAX_SIZE:
            for i in range(self.size - self.MAX_SIZE):
                try:
                    self.safe_data.remove(i)
                except ValueError:
                    self.unsafe_data.remove(i)

            self.safe_data = [i - (self.size - self.MAX_SIZE) for i in self.safe_data]
            self.unsafe_data = [i - (self.size - self.MAX_SIZE) for i in self.unsafe_data]
            del self._data[:self.size - self.MAX_SIZE]  # remove oldest data

    def clear(self):
        self._data.clear()
        self.safe_data = []
        self.unsafe_data = []

    def sample(self, n: int, m: int = 1, balanced_sampling: bool = False) -> List[Data]:
        """
        Sample at random segments of trajectory from buffer.
        Each segment is selected as a symmetric ball w.r.t. randomly sampled data points
        (apart from data points at beginning or end)

        Parameters
        ----------
        n: int,
            number of sample segments
        m: int,
            maximal length of each sampled trajectory segment
        balanced_sampling: bool,
            balance the samples from safe states and unsafe states
        """
        assert self.size >= max(n, m)
        data_list = []
        if not balanced_sampling:
            index = np.sort(np.random.randint(0, self.size, n))

        else:
            index_unsafe, index_safe = [], []
            if len(self.unsafe_data) > 0:
                index_unsafe = random.choices(self.unsafe_data, k=n // 2)
            if len(self.unsafe_data) < 0:
                index_safe = random.choices(self.safe_data, k=n // 2)
            index = sorted(index_safe + index_unsafe)

        ub = 0
        for i in index:
            lb = max(i - m // 2, ub)  # max with ub avoids replicas of the same graph in data_list
            ub = min(i + m // 2 + 1, self.size)
            data_list.extend(self._data[lb:ub])

        return data_list

# class Buffer:
#
#     def __init__(self):
#         self._keys = ('data', 'action')
#         self._data = {
#             'data': [],
#             'action': []
#         }
#
#     def __getitem__(self, item: str):
#         return self._data[item]
#
#     @property
#     def keys(self) -> Tuple[str, ...]:
#         return tuple(self._keys)
#
#     def append(self, key: str, data: Any):
#         assert key in self.keys
#         if len(self._data[key]) > 0:
#             assert isinstance(data, type(self._data[key][-1]))
#         self._data[key].append(data)
#         return self
#
#     def merge(self, other):
#         assert self.keys == other.keys
#         for key in self.keys:
#             assert isinstance(self._data[key][-1], type(other[key][-1]))
#             self._data[key] += other[key]
#         return self
#
#     def clear(self):
#         for key in self.keys:
#             self._data[key].clear()
#
#     def sample(self, key: str, n: int) -> Tuple[Data]:
#         assert key in self.keys
#         length = len(self._data[key])
#         index = np.random.randint(0, length, n)
#         return self._data[key][index]
