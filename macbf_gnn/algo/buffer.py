import numpy as np

from typing import Tuple, Any, Union, Optional, List
from torch_geometric.data import Data


class Buffer:
    # todo: sample segments of trajectory to retain temporal information

    def __init__(self):
        self._data = []  # todo: add maximum number of states

    def append(self, data: Data):
        self._data.append(data)

    @property
    def data(self) -> List[Data]:
        return self._data

    @property
    def size(self) -> int:
        return len(self._data)

    def merge(self, other):
        self._data += other.data

    def clear(self):
        self._data.clear()

    def sample(self, n: int) -> List[Data]:
        assert self.size > 0
        index = np.random.randint(0, self.size, n)
        data_list = []
        for i in index:
            data_list.append(self._data[i])
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
