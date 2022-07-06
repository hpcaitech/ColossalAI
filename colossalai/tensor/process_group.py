import torch
from typing import List, Optional
from colossalai.logging import get_dist_logger


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
                 tp_degree: Optional[int] = None,
                 dp_degree: Optional[int] = None) -> None:
        if not torch.distributed.is_initialized():
            return

        assert torch.distributed.is_initialized(), f"ProcessGroup must be used after distributed initialized"
        if rank is None:
            self._rank = torch.distributed.get_rank()
        else:
            self._rank = rank

        if ranks is None:
            self._rank_list = list(range(torch.distributed.get_world_size()))
        else:
            self._rank_list = ranks

        self._world_size = len(self._rank_list)

        if dp_degree is None and tp_degree is None:
            self._dp_degree = self._world_size
            self._tp_degree = 1

        if dp_degree and not tp_degree:
            self._dp_degree = dp_degree
            assert self._world_size % self._dp_degree == 0, f"DP degree {dp_degree} should be divisible by {self._world_size} hen DP degree is None"
            self._tp_degree = self._world_size // dp_degree

        if not dp_degree and tp_degree:
            self._tp_degree = tp_degree
            assert self._world_size % self._tp_degree == 0, f"TP degree {tp_degree} should be divisible by {self._world_size} when DP degree is None"
            self._dp_degree = self._world_size // tp_degree

        self._tp_rank_list = []
        self._dp_rank_list = []

        for rank_id in range(self._world_size):
            # rank_id and self._rank in the same tp group
            if rank_id % self._tp_degree == self._rank % self._tp_degree:
                self._dp_rank_list.append(rank_id)
            if rank_id // self._tp_degree == self._rank // self._tp_degree:
                self._tp_rank_list.append(rank_id)

        self._tp_process_group = torch.distributed.new_group(ranks=self._tp_rank_list, backend='nccl')
        self._dp_process_group = torch.distributed.new_group(ranks=self._dp_rank_list, backend='nccl')

        self.logger = get_dist_logger('ProcessGroup')
        self.logger.info(
            f'{self._rank} NCCL initialize TP group on {self._tp_rank_list}, DP group on {self._dp_rank_list}')

        self._has_cpu_groups = False

    def set_cpu_groups(self):
        if self.has_cpu_groups:
            return
        self.logger.info(
            f'{self._rank} Gloo initialize TP group on {self._tp_rank_list}, DP group on {self._dp_rank_list}')
        self._cpu_tp_process_group = torch.distributed.new_group(ranks=self._tp_rank_list, backend='gloo')
        self._cpu_dp_process_group = torch.distributed.new_group(ranks=self._dp_rank_list, backend='gloo')

    @property
    def has_cpu_groups(self):
        return self._has_cpu_groups

    def __eq__(self, obj: 'ProcessGroup') -> bool:
        if not isinstance(obj, ProcessGroup):
            return False
        if self._rank != obj._rank:
            assert False
        if self._rank_list != obj._rank_list:
            assert False
        if self._tp_rank_list != obj._tp_rank_list:
            assert False
        if self._dp_rank_list != obj._dp_rank_list:
            assert False
        if self._tp_degree != obj._tp_degree:
            return False
        if self._dp_degree != obj._dp_degree:
            return False
        return True

    def rank(self):
        return self._rank

    def world_size(self):
        return self._world_size

    def tp_local_rank(self):
        return self._rank % self._tp_degree

    def dp_local_rank(self):
        return self._rank // self._tp_degree

    def dp_world_size(self):
        return len(self._dp_rank_list)

    def tp_world_size(self):
        return len(self._tp_rank_list)

    def dp_process_group(self):
        return self._dp_process_group

    def tp_process_group(self):
        return self._tp_process_group

    def cpu_dp_process_group(self):
        return self._cpu_dp_process_group

    def cpu_tp_process_group(self):
        return self._cpu_tp_process_group
