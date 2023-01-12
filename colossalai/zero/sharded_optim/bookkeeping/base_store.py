from colossalai.tensor import ProcessGroup


class BaseStore:

    def __init__(self, pg: ProcessGroup):
        self._world_size = pg.dp_world_size()
        self._local_rank = pg.dp_local_rank()

    @property
    def world_size(self):
        return self._world_size

    @property
    def local_rank(self):
        return self._local_rank
