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

    details of args and kwargs:
    :param rank: The rank of current process
    :param world_size: Size of whole communication world
    :param config: Running configuration
    :param data_parallel_size: Size of data parallel
    :param pipeline_parallel_size: Size of pipeline parallel
    :param tensor_parallel_size: Size of tensor parallel

    :type rank: int
    :type world_size: int
    :type config: Config
    :type data_parallel_size: int
    :type pipeline_parallel_size: int
    :type tensor_parallel_size: int
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
        group_world_size = None
        mode = ParallelMode.DATA

        for i in range(self.num_data_parallel_group):
            ranks = [i + j * self.num_data_parallel_group for j in range(self.data_parallel_size)]
            group = dist.new_group(ranks)

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, ranks_in_group, mode
