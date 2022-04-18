import math

import torch.distributed as dist

from colossalai.registry import DIST_GROUP_INITIALIZER
from .process_group_initializer import ProcessGroupInitializer
from ..parallel_mode import ParallelMode
from colossalai.global_variables import tensor_parallel_env as env


def _check_summa_env_var(summa_dim):
    # check environment variable for SUMMA
    env_summa_dim = env.summa_dim

    if env_summa_dim:
        assert int(env_summa_dim) == summa_dim, \
            'SUMMA_DIM has been set in the current environment and ' \
            'does not match with the value passed to this initialized'
    else:
        env.summa_dim = summa_dim


class Initializer_2D_Row(ProcessGroupInitializer):
    """2d tensor parallel initialization among rows.

    Args:
        num_group (int): The number of all tensor groups.
        summa_dim (int): The dimension of SUMMA.
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """

    def __init__(self, num_group, summa_dim, *args, **kwargs):
        super(Initializer_2D_Row, self).__init__(*args, **kwargs)
        self.num_group = num_group
        self.summa_dim = summa_dim

    def init_dist_group(self):
        """Initialize 2D tensor row parallel groups, and assign local_ranks and groups to each gpu.
        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                2D tensor row parallelism's information in a tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_2D_ROW

        for i in range(self.num_group):
            for j in range(self.summa_dim):
                ranks = [i * self.tensor_parallel_size + j * self.summa_dim + k for k in range(self.summa_dim)]
                group = dist.new_group(ranks)
                group_cpu = dist.new_group(ranks, backend='gloo') if dist.get_backend() != 'gloo' else group

                if self.rank in ranks:
                    local_rank = ranks.index(self.rank)
                    group_world_size = len(ranks)
                    process_group = group
                    cpu_group = group_cpu
                    ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_2D_Col(ProcessGroupInitializer):
    """2d tensor parallel initialization among cols.

    Args:
        num_group (int): The number of all tensor groups.
        summa_dim (int): The dimension of SUMMA.
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """

    def __init__(self, num_group, summa_dim, *args, **kwargs):
        super(Initializer_2D_Col, self).__init__(*args, **kwargs)
        self.num_group = num_group
        self.summa_dim = summa_dim

    def init_dist_group(self):
        """Initialize 2D tensor row parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                2D tensor col parallelism's information in a tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_2D_COL

        for i in range(self.num_group):
            for j in range(self.summa_dim):
                ranks = [i * self.tensor_parallel_size + j + k * self.summa_dim for k in range(self.summa_dim)]
                group = dist.new_group(ranks)
                group_cpu = dist.new_group(ranks, backend='gloo') if dist.get_backend() != 'gloo' else group

                if self.rank in ranks:
                    local_rank = ranks.index(self.rank)
                    group_world_size = len(ranks)
                    process_group = group
                    cpu_group = group_cpu
                    ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


@DIST_GROUP_INITIALIZER.register_module
class Initializer_2D(ProcessGroupInitializer):
    """
    Serve as the single entry point to 2D parallel initialization.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.summa_dim = int(math.sqrt(self.tensor_parallel_size))

        assert self.tensor_parallel_size == self.summa_dim ** 2, \
            "2D summa dim should equal to tensor parallel size ^ 0.5"
        _check_summa_env_var(self.summa_dim)

        self.col_initializer = Initializer_2D_Col(self.num_group, self.summa_dim, *args, **kwargs)
        self.row_initializer = Initializer_2D_Row(self.num_group, self.summa_dim, *args, **kwargs)

    def init_dist_group(self):
        """Initialize 2D tensor row and col parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            List[Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode)]:
                2D tensor parallelism's information in a list of tuples.
        """
        parallel_setting = [self.row_initializer.init_dist_group(), self.col_initializer.init_dist_group()]
        return parallel_setting
