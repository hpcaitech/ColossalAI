#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.distributed as dist
from colossalai.registry import DIST_GROUP_INITIALIZER
from .process_group_initializer import ProcessGroupInitializer
from ..parallel_mode import ParallelMode


@DIST_GROUP_INITIALIZER.register_module
class Initializer_Model(ProcessGroupInitializer):
    """A ProcessGroupInitializer for model parallelism (model parallel group contains pipeline and tensor parallel
    groups).

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
        self.model_parallel_size = self.tensor_parallel_size * self.pipeline_parallel_size
        self.num_group = self.world_size // self.model_parallel_size

    def init_dist_group(self):
        """Initialize model parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                A Model parallelism's information tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.MODEL

        for i in range(self.num_group):
            ranks = [i * self.model_parallel_size + j for j in range(self.model_parallel_size)]
            group = dist.new_group(ranks)
            group_cpu = dist.new_group(ranks, backend='gloo') if dist.get_backend() != 'gloo' else group

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode
