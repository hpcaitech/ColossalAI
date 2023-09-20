#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch.distributed as dist

from colossalai.legacy.registry import DIST_GROUP_INITIALIZER

from ..parallel_mode import ParallelMode
from .initializer_tensor import Initializer_Tensor
from .process_group_initializer import ProcessGroupInitializer


@DIST_GROUP_INITIALIZER.register_module
class Initializer_Sequence_DP(ProcessGroupInitializer):
    """A ProcessGroupInitializer for sequence parallelism all-reduce.

    In Sequence Parallelism, each GPU holds the full copy of model weights,
    thus, gradient all-reduce occurs across all processes in the same pipeline stage

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
        self.dp_size = self.world_size // self.pipeline_parallel_size
        self.num_group = self.pipeline_parallel_size

    def init_dist_group(self):
        """Initialize Sequence Parallel process groups used for gradient all-reduce.

        Returns:
            Tuple: A tuple (local_rank, group_world_size, process_group, ranks_in_group, mode).
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.SEQUENCE_DP

        for i in range(self.num_group):
            ranks = [i * self.dp_size + j for j in range(self.dp_size)]
            group = dist.new_group(ranks)
            group_cpu = dist.new_group(ranks, backend="gloo") if dist.get_backend() != "gloo" else group

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


@DIST_GROUP_INITIALIZER.register_module
class Initializer_Sequence(ProcessGroupInitializer):
    """A ProcessGroupInitializer for sequence parallelism.

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
        # reuse tensor parallel initializer code
        self._sequence_initializer = Initializer_Tensor(*args, **kwargs)
        self._sequence_dp_initializer = Initializer_Sequence_DP(*args, **kwargs)

    def init_dist_group(self):
        """Initialize Sequence parallel process groups and assign local_ranks and groups to each gpu.

        Sequence parallelism requires 2 process groups. The first is for model forward where several processes
        exchange partial query, key and value embedding to compute self attention values. The second is for
        all-reduce to synchronize the model parameters.

        Returns:
            List[Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode)]:
                A Sequence parallelism's information in list of tuples.
        """

        parallel_setting = []

        (
            local_rank,
            group_world_size,
            process_group,
            cpu_group,
            ranks_in_group,
            mode,
        ) = self._sequence_initializer.init_dist_group()
        # change mode to sequence
        mode = ParallelMode.SEQUENCE

        parallel_setting.append((local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode))
        parallel_setting.append(self._sequence_dp_initializer.init_dist_group())
        return parallel_setting
