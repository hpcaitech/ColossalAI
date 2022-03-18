#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch import distributed as dist

from colossalai.registry import DIST_GROUP_INITIALIZER
from .process_group_initializer import ProcessGroupInitializer
from ..parallel_mode import ParallelMode


@DIST_GROUP_INITIALIZER.register_module
class Initializer_Pipeline(ProcessGroupInitializer):
    """A ProcessGroupInitializer for pipeline parallelism.

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
        self.data_group_size = self.world_size // self.data_parallel_size
        self.pipeline_stage_size = self.data_group_size // self.pipeline_parallel_size

    def init_dist_group(self):
        """Initialize pipeline parallel groups, and assign local_ranks and groups to each gpu.

        :return: Pipeline parallelism's information
        :rtype: list of Tuples (local_rank, group_world_size, process_group, ranks_in_group, mode)
        """
        dist_settings = list()
        for i in range(self.data_parallel_size):
            for j in range(self.pipeline_stage_size):
                pipe_ranks = list(
                    range(i * self.data_group_size + j,
                          (i + 1) * self.data_group_size,
                          self.pipeline_stage_size))
                pipe_group_size = len(pipe_ranks)
                pipe_group = dist.new_group(pipe_ranks)

                if self.rank in pipe_ranks:
                    local_rank = pipe_ranks.index(self.rank)
                    group_world_size = pipe_group_size
                    process_group = pipe_group
                    ranks_in_group = pipe_ranks
                    dist_settings.append(
                        tuple((local_rank, group_world_size,
                               process_group, ranks_in_group,
                               ParallelMode.PIPELINE)))

        return dist_settings
