#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch import distributed as dist

from colossalai.legacy.registry import DIST_GROUP_INITIALIZER

from ..parallel_mode import ParallelMode
from .process_group_initializer import ProcessGroupInitializer


@DIST_GROUP_INITIALIZER.register_module
class Initializer_Data(ProcessGroupInitializer):
    """A ProcessGroupInitializer for data parallelism.

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
        self.num_data_parallel_group = self.world_size // self.data_parallel_size

    def init_dist_group(self):
        """Initialize data parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                A Data parallelism's information tuple.
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
            group_cpu = dist.new_group(ranks, backend="gloo") if dist.get_backend() != "gloo" else group

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode
