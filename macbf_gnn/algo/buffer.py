import numpy as np

from typing import Tuple, Any, Union, Optional, List
from torch_geometric.data import Data


class Buffer:
    def __init__(self):
        self._data = []
        self.MAX_SIZE = 2e5

    def append(self, data: Data):
        self._data.append(data)
        if len(self._data) > self.MAX_SIZE:
            del self._data[0]

    @property
    def data(self) -> List[Data]:
        return self._data

    @property
    def size(self) -> int:
        return len(self._data)

    def merge(self, other):
        self._data += other.data
        if len(self._data) > self.MAX_SIZE:
            del self._data[0:len(self._data)-self.MAX_SIZE]

    def clear(self):
        self._data.clear()

    def sample(self, n: int, m: int=1) -> List[Data]:
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
        """
        assert self.size >= max(n, m)
        index = np.random.randint(0, self.size, n)
        data_list = []
        ub = 0
        for i in index:
            lb = max(i - m // 2, ub)  # max with ub avoids replicas of the same graph
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
