import torch
from typing import List, Optional
from colossalai.logging import get_dist_logger
from colossalai.context.singleton_meta import SingletonMeta


class PyTorchProcessGroupDict(metaclass=SingletonMeta):

    def __init__(self):
        # distributed settings
        self.dict = {}

    def get(self, rank_list: List[int], backend: str = 'nccl'):
        """Reuse Pytorch ProcessGroup when such a group is initialized
        """
        rank_tuple = tuple(rank_list)
        # we need to convert the passed list to a tuple
        # since List is unhashable
        pg_key = (backend, rank_tuple)

        if pg_key not in self.dict:

            self.logger = get_dist_logger('ProcessGroup')
            self.logger.info(f'NCCL initialize TP group on {rank_list}', ranks=[0])

            self.dict[pg_key] = torch.distributed.new_group(ranks=rank_list, backend=backend)
        return self.dict[pg_key]


PYTORCHPGDICT_ = PyTorchProcessGroupDict()


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
            self._rank_list.sort()    # ensure that the list is in order

        self._rank_idx = self._rank_list.index(self._rank)
        self._world_size = len(self._rank_list)

        if dp_degree is None and tp_degree is None:
            self._dp_degree = self._world_size
            self._tp_degree = 1
        elif dp_degree and not tp_degree:
            self._dp_degree = dp_degree
            assert self._world_size % self._dp_degree == 0, f"DP degree {dp_degree} should be divisible by {self._world_size} hen DP degree is None"
            self._tp_degree = self._world_size // dp_degree
        elif not dp_degree and tp_degree:
            self._tp_degree = tp_degree
            assert self._world_size % self._tp_degree == 0, f"TP degree {tp_degree} should be divisible by {self._world_size} when DP degree is None"
            self._dp_degree = self._world_size // tp_degree
        else:
            self._dp_degree = dp_degree
            self._tp_degree = tp_degree
            assert self._dp_degree * self._tp_degree == self._world_size, \
                f"the world size {self._world_size} should equals to the product of DP degree {self._dp_degree}" \
                f"and TP degree {self._tp_degree}"

        self._tp_rank_list = []
        self._dp_rank_list = []

        for idx, rank_id in enumerate(self._rank_list):
            # idx and self._rank_idx in the same tp group
            if idx % self._tp_degree == self._rank_idx % self._tp_degree:
                self._dp_rank_list.append(rank_id)
            if idx // self._tp_degree == self._rank_idx // self._tp_degree:
                self._tp_rank_list.append(rank_id)

        self._tp_process_group = PYTORCHPGDICT_.get(self._tp_rank_list, 'nccl')
        self._dp_process_group = PYTORCHPGDICT_.get(self._dp_rank_list, 'nccl')

        self._has_cpu_groups = False
        self._cpu_dp_process_group = None
        self._cpu_tp_process_group = None

    def set_cpu_groups(self):
        if self.has_cpu_groups:
            return
        self.logger.info(
            f'{self._rank} Gloo initialize TP group on {self._tp_rank_list}, DP group on {self._dp_rank_list}')
        self._cpu_tp_process_group = PYTORCHPGDICT_.get(self._tp_rank_list, 'gloo')
        self._cpu_dp_process_group = PYTORCHPGDICT_.get(self._dp_rank_list, 'gloo')

    @property
    def has_cpu_groups(self):
        return self._has_cpu_groups

    def __repr__(self):
        return "ProcessGroup:\n\tRank: {}, World size: {}, DP degree: {}, TP degree: {}\n\tRanks in group: {}".\
            format(self._rank, self._world_size, self._dp_degree, self._tp_degree, self._rank_list)

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
