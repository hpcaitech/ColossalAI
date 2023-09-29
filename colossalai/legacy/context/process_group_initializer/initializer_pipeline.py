#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch import distributed as dist

from colossalai.legacy.registry import DIST_GROUP_INITIALIZER

from ..parallel_mode import ParallelMode
from .process_group_initializer import ProcessGroupInitializer


@DIST_GROUP_INITIALIZER.register_module
class Initializer_Pipeline(ProcessGroupInitializer):
    """A ProcessGroupInitializer for pipeline parallelism.

    Args:
        rank (int): The rank of current process
        world_size (int): Size of whole communication world
        config (Config): Running configuration
        data_parallel_size (int): Size of data parallel
        pipeline_parallel_size (int): Size of pipeline parallel
        tensor_parallel_size (int): Size of tensor parallel
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_group_size = self.world_size // self.data_parallel_size
        self.pipeline_stage_size = self.data_group_size // self.pipeline_parallel_size

    def init_dist_group(self):
        """Initialize pipeline parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            List[Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode)]:
                A Pipeline parallelism's information in list of tuples.
        """
        dist_settings = list()
        for i in range(self.data_parallel_size):
            for j in range(self.pipeline_stage_size):
                pipe_ranks = list(
                    range(i * self.data_group_size + j, (i + 1) * self.data_group_size, self.pipeline_stage_size)
                )
                pipe_group_size = len(pipe_ranks)
                pipe_group = dist.new_group(pipe_ranks)
                group_cpu = dist.new_group(pipe_ranks, backend="gloo") if dist.get_backend() != "gloo" else pipe_group

                if self.rank in pipe_ranks:
                    local_rank = pipe_ranks.index(self.rank)
                    group_world_size = pipe_group_size
                    process_group = pipe_group
                    cpu_group = group_cpu
                    ranks_in_group = pipe_ranks
                    dist_settings.append(
                        tuple(
                            (
                                local_rank,
                                group_world_size,
                                process_group,
                                cpu_group,
                                ranks_in_group,
                                ParallelMode.PIPELINE,
                            )
                        )
                    )

        return dist_settings
