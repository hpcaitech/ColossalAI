#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.distributed as dist
from colossalai.global_variables import tensor_parallel_env as env
from colossalai.registry import DIST_GROUP_INITIALIZER

from ..parallel_mode import ParallelMode
from .process_group_initializer import ProcessGroupInitializer


@DIST_GROUP_INITIALIZER.register_module
class Initializer_1D(ProcessGroupInitializer):
    '''A ProcessGroupInitializer for 1d tensor parallelism.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = self.world_size // self.tensor_parallel_size

    def init_dist_group(self):
        """Initialize 1D tensor parallel groups, and assign local_ranks and groups to each gpu.
        :return: (local_rank, group_world_size, process_group, ranks_in_group, mode)
        :rtype: Tuple
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_1D
        env.parallel_input_1d = False

        for i in range(self.num_group):
            ranks = [i * self.tensor_parallel_size + j for j in range(self.tensor_parallel_size)]
            group = dist.new_group(ranks)

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, ranks_in_group, mode
