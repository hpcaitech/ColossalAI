from typing import Tuple, Union

import numpy as np
import torch.distributed as dist
from torch.distributed import ProcessGroup


class BaseStore:
    def __init__(self, torch_pg: Union[ProcessGroup, Tuple[ProcessGroup, ...]]):
        if isinstance(torch_pg, tuple):
            self.sizes = [dist.get_world_size(group=pg) for pg in torch_pg]
            self._world_size = int(np.prod(self.sizes))
            self._local_rank = np.ravel_multi_index(tuple(dist.get_rank(group=pg) for pg in torch_pg), self.sizes)
        else:
            self._world_size = dist.get_world_size(group=torch_pg)
            self._local_rank = dist.get_rank(group=torch_pg)
            self.sizes = [self._world_size]
        self.torch_pg = torch_pg

    @property
    def world_size(self):
        return self._world_size

    @property
    def local_rank(self):
        return self._local_rank
