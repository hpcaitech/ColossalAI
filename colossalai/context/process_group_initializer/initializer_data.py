#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch import distributed as dist

from colossalai.registry import DIST_GROUP_INITIALIZER
from .process_group_initializer import ProcessGroupInitializer
from ..parallel_mode import ParallelMode


@DIST_GROUP_INITIALIZER.register_module
class Initializer_Data(ProcessGroupInitializer):
    """A ProcessGroupInitializer for data parallelism.

    :param args: Args used to initialize ProcessGroupInitializer
    :param kwargs: Kwargs used to initialize ProcessGroupInitializer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_data_parallel_group = self.world_size // self.data_parallel_size

    def init_dist_group(self):
        """Initialize data parallel groups, and assign local_ranks and groups to each gpu.

        :return: Data parallelism's information
        :rtype: Tuple(local_rank, group_world_size, process_group, ranks_in_group, mode)
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.DATA

        for i in range(self.num_data_parallel_group):
            ranks = [i + j * self.num_data_parallel_group for j in range(self.data_parallel_size)]
            group = dist.new_group(ranks)
            group_cpu = dist.new_group(ranks, backend='gloo') if dist.get_backend() != 'gloo' else group

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode
