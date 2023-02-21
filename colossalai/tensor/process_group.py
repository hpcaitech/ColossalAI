from typing import List, Optional

import torch

from colossalai.context.singleton_meta import SingletonMeta
from colossalai.logging import get_dist_logger


class PyTorchProcessGroupDict(metaclass=SingletonMeta):

    def __init__(self):
        # distributed settings
        # use this dict to record all Pytorch ProcessGroups
        self.dict = {}
        # set a distributed logger
        self.logger = get_dist_logger('ProcessGroup')

    def log_pg_init(self, rank_list: List[int], backend: str):
        str_list = ["Pytorch ProcessGroup Init:"]
        str_list.append(f"backend: {backend}")
        str_list.append(f"ranks: {rank_list}")
        self.logger.info("\n\t".join(str_list), ranks=[0])

    def get(self, rank_list: List[int], backend: str = 'nccl'):
        """Reuse Pytorch ProcessGroup when such a group is initialized
        """
        # we need to convert the passed list to a tuple
        # since List is unhashable
        processgroup_key = (backend, tuple(rank_list))
        if processgroup_key not in self.dict:
            self.log_pg_init(rank_list=rank_list, backend=backend)
            self.dict[processgroup_key] = torch.distributed.new_group(ranks=rank_list, backend=backend)
        return self.dict[processgroup_key]


PYTORCHPGDICT_ = PyTorchProcessGroupDict()


class ProcessGroup:
    """ProcessGroup
    Process Group indicates how processes are organized in groups for parallel execution using Tensor Parallelism and Data Parallelism.

    NOTE, the ProcessGroup must be used after `torch.distributed.initialize()`


    Args:
        rank: the global rank of the current process.
        ranks: List[int], a list of rank id belongings to this process group.
        backend: str, the backend of the process group.
        tp_degree: Optional[int], tensor parallelism degree. How many processes are inside a tp process group. default None means 1.
        dp_degree: Optional[int], data parallelism degree. How many processes are inside a dp process group. . default None means len(ranks).
    """

    def __init__(self,
                 rank: Optional[int] = None,
                 ranks: Optional[List[int]] = None,
                 tp_degree: Optional[int] = None,
                 dp_degree: Optional[int] = None) -> None:
        if not torch.distributed.is_initialized():
            self.is_init = False
            return

        assert torch.distributed.is_initialized(), f"ProcessGroup must be used after distributed initialized"

        self._rank = torch.distributed.get_rank()
        if rank is not None:
            assert self._rank == rank    # make sure that the global rank is correct

        if ranks is None:
            self._rank_list = list(range(torch.distributed.get_world_size()))
        else:
            self._rank_list = ranks
            self._rank_list.sort()    # ensure that the list is in order

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

        self._tp_rank_list = None
        self._dp_rank_list = None

        for i in range(self._dp_degree):
            i_tp_list = [self._rank_list[i * self._tp_degree + j] for j in range(self._tp_degree)]
            PYTORCHPGDICT_.get(i_tp_list, 'nccl')
            if self._rank in i_tp_list:
                self._tp_rank_list = i_tp_list

        for j in range(self._tp_degree):
            j_dp_list = [self._rank_list[i * self._tp_degree + j] for i in range(self._dp_degree)]
            PYTORCHPGDICT_.get(j_dp_list, 'nccl')
            if self._rank in j_dp_list:
                self._dp_rank_list = j_dp_list

        self._has_cpu_groups = False
        self.is_init = True

    def set_cpu_groups(self):
        """set_cpu_groups
        Initialize Pytorch process groups for cpu communications.
        """
        if self.has_cpu_groups:
            return

        for i in range(self._dp_degree):
            i_tp_list = [self._rank_list[i * self._tp_degree + j] for j in range(self._tp_degree)]
            PYTORCHPGDICT_.get(i_tp_list, 'gloo')

        for j in range(self._tp_degree):
            j_dp_list = [self._rank_list[i * self._tp_degree + j] for i in range(self._dp_degree)]
            PYTORCHPGDICT_.get(j_dp_list, 'gloo')

        self._has_cpu_groups = True

    @property
    def has_cpu_groups(self) -> bool:
        """has_cpu_groups
        If cpu groups have been initailized.

        Returns:
            bool: cpu process groups have been initialized or not.
        """
        return self._has_cpu_groups

    def __repr__(self):
        if self.is_init:
            ranks_str = f"ProcessGroup(ranks={self._rank_list},\n"
            personal_str = f"             rank={self._rank}, dp={self._dp_degree}, tp={self._tp_degree})"
            return ranks_str + personal_str
        else:
            return "ProcessGroup not initialized"

    def __eq__(self, obj: 'ProcessGroup') -> bool:
        if not isinstance(obj, ProcessGroup):
            return False
        if self._rank != obj._rank:
            return False
        if self._rank_list != obj._rank_list:
            return False
        if self._tp_rank_list != obj._tp_rank_list:
            return False
        if self._dp_rank_list != obj._dp_rank_list:
            return False
        if self._tp_degree != obj._tp_degree:
            return False
        if self._dp_degree != obj._dp_degree:
            return False
        return True

    def rank(self) -> int:
        """rank

        The current rank in the global process group.

        Returns:
            int: the rank number
        """
        return self._rank

    def ranks_in_group(self) -> List[int]:
        """ranks_in_group

        a list of rank number in in the global process group.

        Returns:
            List[int]: a list of rank number.
        """
        return self._rank_list

    def world_size(self) -> int:
        """world_size

        The world size of the global process group.

        Returns:
            int: world size
        """
        return self._world_size

    def tp_rank_list(self) -> List[int]:
        """tp_rank_list

        the rank list in the TP process group containing the current rank.

        Returns:
            List[int]: the list of rank number.
        """
        return self._tp_rank_list

    def dp_rank_list(self) -> List[int]:
        """dp_rank_list

        the rank list in the DP process group containing the current rank.

        Returns:
            List[int]:  the list of rank number.
        """
        return self._dp_rank_list

    def tp_local_rank(self) -> int:
        """tp_local_rank

        The local rank number in the current TP process group.

        Returns:
            int: tp rank number.
        """
        return self._rank % self._tp_degree

    def dp_local_rank(self) -> int:
        """dp_local_rank

        The local rank number in the current DP process group.

        Returns:
            int: dp rank number.
        """
        return self._rank // self._tp_degree

    def dp_world_size(self) -> int:
        """dp_world_size

        The world size of the current DP process group.

        Returns:
            int: dp world size
        """
        return len(self._dp_rank_list)

    def tp_world_size(self) -> int:
        """tp_world_size

        The world size of the current TP process group.

        Returns:
            int: tp world size
        """
        return len(self._tp_rank_list)

    def dp_process_group(self):
        """dp_process_group

        the pytorch DP process group containing the current rank.

        Returns:
            `torch._C._distributed_c10d.ProcessGroup`: the pytorch DP process group.
        """
        return PYTORCHPGDICT_.get(self._dp_rank_list, 'nccl')

    def tp_process_group(self):
        """tp_process_group

        the pytorch TP process group containing the current rank.

        Returns:
            `torch._C._distributed_c10d.ProcessGroup`: the pytorch TP process group.
        """
        return PYTORCHPGDICT_.get(self._tp_rank_list, 'nccl')

    def cpu_dp_process_group(self):
        """cpu_dp_process_group

        the pytorch CPU DP process group containing the current rank.

        assert failed if cpu process group is not initialized.

        Returns:
            `torch._C._distributed_c10d.ProcessGroup`: the pytorch DP process group.
        """
        assert self._has_cpu_groups
        return PYTORCHPGDICT_.get(self._dp_rank_list, 'gloo')

    def cpu_tp_process_group(self):
        """cpu_tp_process_group

        the pytorch CPU TP process group containing the current rank.

        assert failed if cpu process group is not initialized.

        Returns:
            `torch._C._distributed_c10d.ProcessGroup`: the pytorch TP process group.
        """
        assert self._has_cpu_groups
        return PYTORCHPGDICT_.get(self._tp_rank_list, 'gloo')

    def get_ranks_in_dp(self) -> List[int]:
        """get_ranks_in_dp

        ranks in current dp process group.

        Returns:
            List[int]: a list of rank number.
        """
        return self._dp_rank_list

    def get_ranks_in_tp(self):
        """get_ranks_in_tp

        ranks in current tp process group.

        Returns:
            List[int]: a list of rank number.
        """
        return self._tp_rank_list
