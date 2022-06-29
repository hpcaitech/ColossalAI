import torch
from typing import List, Optional


class ProcessGroup:
    """
    Process Group contains group partition for Tensor Parallel and Data Parallel.
    NOTE, the ProcessGroup must be used after torch.distributed.initialize()
    args:
        rank: the global rank of the current process.
        ranks: List[int], a list of rank id belongings to this process group.
        backend: str, the backend of the process group.
        tp_degree: Optional[int], tensor parallelism degree, default None means 1
        dp_degree: Optional[int], data parallelism degree, default None means len(ranks)
    """

    def __init__(self,
                 rank: Optional[int] = None,
                 ranks: Optional[List[int]] = None,
                 backend: str = 'nccl',
                 tp_degree: Optional[int] = None,
                 dp_degree: Optional[int] = None) -> None:
        assert torch.distributed.is_initialized(), f"ProcessGroup must be used after distributed initialized"
        if rank is None:
            self._rank = torch.distributed.get_rank()
        else:
            self._rank = rank

        if ranks is None:
            self._rank_list = list(range(torch.distributed.get_world_size()))
        else:
            self._rank_list = ranks

        self._backend = backend
        self._world_size = len(self._rank_list)

        if dp_degree is None and tp_degree is None:
            self._dp_degree = self._world_size
            self._tp_degree = 1

        if dp_degree and not tp_degree:
            self._dp_degree = dp_degree
            assert self._world_size % self._dp_degree == 0, f"DP degree {dp_degree} should be divisible by {self._world_size} hen DP degree is None"
            self._tp_degree = self._world_size / dp_degree

        if not dp_degree and tp_degree:
            self._tp_degree = tp_degree
            assert self._world_size % self._tp_degree == 0, f"TP degree {tp_degree} should be divisible by {self._world_size} when DP degree is None"
            self._dp_degree = self._world_size / tp_degree

        self._tp_rank_list = []
        self._dp_rank_list = []

        for rank_id in range(self._world_size):
            # rank_id and self._rank in the same tp group
            if rank_id % self._tp_degree == self._rank % self._tp_degree:
                self._dp_rank_list.append(rank_id)
            if rank_id // self._tp_degree == self._rank // self._tp_degree:
                self._tp_rank_list.append(rank_id)

        self._tp_process_group = torch.distributed.new_group(ranks=self._tp_rank_list, backend=backend)
        self._dp_process_group = torch.distributed.new_group(ranks=self._dp_rank_list, backend=backend)

    def world_size(self):
        return self._world_size

    def dp_world_size(self):
        return len(self._dp_rank_list)

    def tp_world_size(self):
        return len(self._tp_rank_list)

    def dp_process_group(self):
        return self._dp_process_group

    def tp_process_group(self):
        return self._tp_process_group
