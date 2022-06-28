import torch
from typing import List


class ProcessGroup:

    def __init__(self, ranks: List[int], backend: str = 'nccl') -> None:
        self._ranks = ranks
        self._backend = backend
        self._world_size = len(self._ranks)
        assert torch.distributed.is_initialized(), f"ProcessGroup must be used after distributed initialized"
        self._pg = torch.distributed.new_group(ranks=ranks, backend=backend)

    def world_size(self):
        return self._world_size

    def group(self):
        return self._pg
