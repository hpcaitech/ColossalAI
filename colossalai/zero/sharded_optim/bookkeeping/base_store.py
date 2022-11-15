from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc


class BaseStore:

    def __init__(self, dp_parallel_mode=ParallelMode.DATA):
        self._world_size = gpc.get_world_size(dp_parallel_mode)
        self._local_rank = gpc.get_local_rank(dp_parallel_mode)

    @property
    def world_size(self):
        return self._world_size

    @property
    def local_rank(self):
        return self._local_rank
