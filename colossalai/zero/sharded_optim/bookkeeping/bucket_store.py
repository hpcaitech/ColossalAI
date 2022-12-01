from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc

from .base_store import BaseStore


class BucketStore(BaseStore):

    def __init__(self, dp_parallel_mode):
        super().__init__(dp_parallel_mode)
        self._grads = dict()
        self._params = dict()
        self._num_elements_in_bucket = dict()

        self.reset()

    def num_elements_in_bucket(self, reduce_rank: int = None):
        return self._num_elements_in_bucket[reduce_rank]

    def add_num_elements_in_bucket(self, num_elements, reduce_rank: int = None):
        self._num_elements_in_bucket[reduce_rank] += num_elements

    def add_grad(self, tensor, reduce_rank: int = None):
        self._grads[reduce_rank].append(tensor)

    def add_param(self, tensor, reduce_rank: int = None):
        self._params[reduce_rank].append(tensor)

    def reset(self):
        keys = [None] + list(range(self._world_size))
        self._grads = {rank: [] for rank in keys}
        self._params = {rank: [] for rank in keys}
        self._num_elements_in_bucket = {rank: 0 for rank in keys}

    def reset_by_rank(self, reduce_rank=None):
        self._grads[reduce_rank] = []
        self._params[reduce_rank] = []
        self._num_elements_in_bucket[reduce_rank] = 0

    def get_grad(self, reduce_rank: int = None):
        return self._grads[reduce_rank]

    def get_param(self, reduce_rank: int = None):
        return self._params[reduce_rank]
