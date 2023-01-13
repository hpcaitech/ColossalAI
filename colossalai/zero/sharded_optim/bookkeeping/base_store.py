from typing import Optional

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.tensor import ProcessGroup


class BaseStore:

    def __init__(self, pg: Optional[ProcessGroup] = None):
        if isinstance(pg, ProcessGroup):
            self._world_size = pg.dp_world_size()
            self._local_rank = pg.dp_local_rank()
        else:
            self._world_size = gpc.get_world_size(ParallelMode.DATA)
            self._local_rank = gpc.get_local_rank(ParallelMode.DATA)

    @property
    def world_size(self):
        return self._world_size

    @property
    def local_rank(self):
        return self._local_rank
