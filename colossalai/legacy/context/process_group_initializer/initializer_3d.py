#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math

import torch.distributed as dist

from colossalai.legacy.global_variables import tensor_parallel_env as env
from colossalai.legacy.registry import DIST_GROUP_INITIALIZER

from ..parallel_mode import ParallelMode
from .process_group_initializer import ProcessGroupInitializer


def _check_depth_env_var(depth):
    # check global variable
    env_depth = env.depth_3d

    if env_depth:
        assert int(env_depth) == depth, (
            "DEPTH_3D has been set in the current environment and "
            "does not match with the value passed to this initialized"
        )
    else:
        env.depth_3d = depth


class Initializer_3D_Input(ProcessGroupInitializer):
    """3D tensor parallel initialization among input.

    Args:
        num_group (int): The number of all tensor groups.
        depth (int): Depth of 3D parallelism.
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """

    def __init__(self, num_group: int, depth: int, *args):
        super().__init__(*args)
        self.num_group = num_group
        self.depth = depth

    def init_dist_group(self):
        """Initialize 3D tensor parallel groups among input, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                3D tensor parallelism's information among input in a tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_3D_INPUT
        env.input_group_3d = mode

        for h in range(self.num_group):
            for i in range(self.depth):
                for k in range(self.depth):
                    ranks = [h * self.depth**3 + i + self.depth * (j + self.depth * k) for j in range(self.depth)]
                    group = dist.new_group(ranks)
                    group_cpu = dist.new_group(ranks, backend="gloo") if dist.get_backend() != "gloo" else group

                    if self.rank in ranks:
                        local_rank = ranks.index(self.rank)
                        group_world_size = len(ranks)
                        process_group = group
                        cpu_group = group_cpu
                        ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_3D_Weight(ProcessGroupInitializer):
    """3D tensor parallel initialization among weight.

    Args:
        num_group (int): The number of all tensor groups.
        depth (int): Depth of 3D parallelism.
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """

    def __init__(self, num_group: int, depth: int, *args):
        super().__init__(*args)
        self.num_group = num_group
        self.depth = depth

    def init_dist_group(self):
        """Initialize 3D tensor parallel groups among weight, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                3D tensor parallelism's information among weight in a tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_3D_WEIGHT
        env.weight_group_3d = mode

        for h in range(self.num_group):
            for k in range(self.depth):
                for j in range(self.depth):
                    ranks = [h * self.depth**3 + i + self.depth * (j + self.depth * k) for i in range(self.depth)]
                    group = dist.new_group(ranks)
                    group_cpu = dist.new_group(ranks, backend="gloo") if dist.get_backend() != "gloo" else group

                    if self.rank in ranks:
                        local_rank = ranks.index(self.rank)
                        group_world_size = len(ranks)
                        process_group = group
                        cpu_group = group_cpu
                        ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_3D_Output(ProcessGroupInitializer):
    """3D tensor parallel initialization among output.

    Args:
        num_group (int): The number of all tensor groups.
        depth (int): Depth of 3D parallelism.
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """

    def __init__(self, num_group: int, depth: int, *args):
        super().__init__(*args)
        self.num_group = num_group
        self.depth = depth

    def init_dist_group(self):
        """Initialize 3D tensor parallel groups among output, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                3D tensor parallelism's information among output in a tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_3D_OUTPUT
        env.output_group_3d = mode

        for h in range(self.num_group):
            for i in range(self.depth):
                for j in range(self.depth):
                    ranks = [h * self.depth**3 + i + self.depth * (j + self.depth * k) for k in range(self.depth)]
                    group = dist.new_group(ranks)
                    group_cpu = dist.new_group(ranks, backend="gloo") if dist.get_backend() != "gloo" else group

                    if self.rank in ranks:
                        local_rank = ranks.index(self.rank)
                        group_world_size = len(ranks)
                        process_group = group
                        cpu_group = group_cpu
                        ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_3D_InputxWeight(ProcessGroupInitializer):
    """3D tensor parallel initialization among input.

    Args:
        num_group (int): The number of all tensor groups.
        depth (int): Depth of 3D parallelism.
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """

    def __init__(self, num_group: int, depth: int, *args):
        super().__init__(*args)
        self.num_group = num_group
        self.depth = depth

    def init_dist_group(self):
        """Initialize 3D tensor parallel groups among input, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                3D tensor parallelism's information among input in a tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_3D_INPUT_X_WEIGHT
        env.input_x_weight_group_3d = mode

        for h in range(self.num_group):
            for k in range(self.depth):
                ranks = [
                    h * self.depth**3 + i + self.depth * (j + self.depth * k)
                    for j in range(self.depth)
                    for i in range(self.depth)
                ]
                group = dist.new_group(ranks)
                group_cpu = dist.new_group(ranks, backend="gloo") if dist.get_backend() != "gloo" else group

                if self.rank in ranks:
                    local_rank = ranks.index(self.rank)
                    group_world_size = len(ranks)
                    process_group = group
                    cpu_group = group_cpu
                    ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_3D_OutputxWeight(ProcessGroupInitializer):
    """3D tensor parallel initialization among input.

    Args:
        num_group (int): The number of all tensor groups.
        depth (int): Depth of 3D parallelism.
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """

    def __init__(self, num_group: int, depth: int, *args):
        super().__init__(*args)
        self.num_group = num_group
        self.depth = depth

    def init_dist_group(self):
        """Initialize 3D tensor parallel groups among input, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                3D tensor parallelism's information among input in a tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_3D_OUTPUT_X_WEIGHT
        env.output_x_weight_group_3d = mode

        for h in range(self.num_group):
            for j in range(self.depth):
                ranks = [
                    h * self.depth**3 + i + self.depth * (j + self.depth * k)
                    for k in range(self.depth)
                    for i in range(self.depth)
                ]
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
class Initializer_3D(ProcessGroupInitializer):
    """Serve as the single entry point to 3D parallel initialization.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.depth = round(math.pow(self.tensor_parallel_size, 1 / 3))
        assert (
            self.tensor_parallel_size == self.depth**3
        ), f"3D depth ({self.depth}) if not cube root of tensor parallel size ({self.tensor_parallel_size})"
        _check_depth_env_var(self.depth)

        self.input_initializer = Initializer_3D_Input(self.num_group, self.depth, *args)
        self.weight_initializer = Initializer_3D_Weight(self.num_group, self.depth, *args)
        self.output_initializer = Initializer_3D_Output(self.num_group, self.depth, *args)
        self.input_x_weight_initializer = Initializer_3D_InputxWeight(self.num_group, self.depth, *args)
        self.output_x_weight_initializer = Initializer_3D_OutputxWeight(self.num_group, self.depth, *args)

    def init_dist_group(self):
        """Initialize 3D tensor parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            List[Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode)]:
                Whole 3D tensor parallelism's information in a list of tuples.
        """
        parallel_setting = [
            self.input_initializer.init_dist_group(),
            self.weight_initializer.init_dist_group(),
            self.output_initializer.init_dist_group(),
            self.input_x_weight_initializer.init_dist_group(),
            self.output_x_weight_initializer.init_dist_group(),
        ]
        return parallel_setting
