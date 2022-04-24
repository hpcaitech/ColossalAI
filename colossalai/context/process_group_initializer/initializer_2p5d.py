#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math

import torch.distributed as dist
from colossalai.context import Config
from colossalai.global_variables import tensor_parallel_env as env
from colossalai.registry import DIST_GROUP_INITIALIZER

from ..parallel_mode import ParallelMode
from .process_group_initializer import ProcessGroupInitializer


def _check_tesseract_env_var(tesseract_dim: int, tesseract_dep: int):
    # check global variable for TESSERACT
    env_tesseract_dim = env.tesseract_dim
    env_tesseract_dep = env.tesseract_dep

    if env_tesseract_dim and env_tesseract_dep:
        assert int(env_tesseract_dim) == tesseract_dim, \
            'TESSERACT_DIM has been set in the current environment and ' \
            'does not match with the value passed to this initialized'
        assert int(env_tesseract_dep) == tesseract_dep, \
            'TESSERACT_DEP has been set in the current environment and ' \
            'does not match with the value passed to this initialized'
    else:
        env.tesseract_dim = tesseract_dim
        env.tesseract_dep = tesseract_dep


# i row j col k dep
class Initializer_2p5D_ROW(ProcessGroupInitializer):
    """2.5d tensor parallel initialization among rows.

    Args:
        tesseract_dim (int): The dimension of tesseract.
        tesseract_dep (int): The dimension of depth.
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """

    def __init__(self, tesseract_dim: int, tesseract_dep: int, *args):
        super(Initializer_2p5D_ROW, self).__init__(*args)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.tesseract_dep = tesseract_dep
        self.tesseract_dim = tesseract_dim
        assert self.tensor_parallel_size == self.tesseract_dim ** 2 * self.tesseract_dep, \
            "Tensor parallel size should be depth * dim ** 2 in 2.5D parallel"

    def init_dist_group(self):
        """Initialize 2.5D tensor row parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                2.5D tensor row parallelism's information in a tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_2P5D_ROW

        for h in range(self.num_group):
            for j in range(self.tesseract_dim):
                for k in range(self.tesseract_dep):
                    ranks = [
                        h * self.tensor_parallel_size + i + self.tesseract_dim * (j + self.tesseract_dim * k)
                        for i in range(self.tesseract_dim)
                    ]
                    group = dist.new_group(ranks)
                    group_cpu = dist.new_group(ranks, backend='gloo') if dist.get_backend() != 'gloo' else group

                    if self.rank in ranks:
                        local_rank = ranks.index(self.rank)
                        group_world_size = len(ranks)
                        process_group = group
                        cpu_group = group_cpu
                        ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_2p5D_Col(ProcessGroupInitializer):
    """2.5d tensor parallel initialization among cols.

    Args:
        tesseract_dim (int): The dimension of tesseract.
        tesseract_dep (int): The dimension of depth.
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """

    def __init__(self, tesseract_dim: int, tesseract_dep: int, *args):
        super(Initializer_2p5D_Col, self).__init__(*args)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.tesseract_dep = tesseract_dep
        self.tesseract_dim = tesseract_dim

    def init_dist_group(self):
        """Initialize 2.5D tensor col parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                2.5D tensor col parallelism's information in a tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_2P5D_COL

        for h in range(self.num_group):
            for i in range(self.tesseract_dim):
                for k in range(self.tesseract_dep):
                    ranks = [
                        h * self.tensor_parallel_size + i + self.tesseract_dim * (j + self.tesseract_dim * k)
                        for j in range(self.tesseract_dim)
                    ]
                    group = dist.new_group(ranks)
                    group_cpu = dist.new_group(ranks, backend='gloo') if dist.get_backend() != 'gloo' else group

                    if self.rank in ranks:
                        local_rank = ranks.index(self.rank)
                        group_world_size = len(ranks)
                        process_group = group
                        cpu_group = group_cpu
                        ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_2p5D_Dep(ProcessGroupInitializer):
    """2.5D tensor parallel initialization among depths.

    Args:
        tesseract_dim (int): The dimension of tesseract.
        tesseract_dep (int): The dimension of depth.
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """

    def __init__(self, tesseract_dim: int, tesseract_dep: int, *args):
        super(Initializer_2p5D_Dep, self).__init__(*args)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.tesseract_dep = tesseract_dep
        self.tesseract_dim = tesseract_dim

    def init_dist_group(self):
        """Initialize 2.5D tensor depth parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                2.5D tensor depth parallelism's information in a tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_2P5D_DEP

        for h in range(self.num_group):
            for i in range(self.tesseract_dim):
                for j in range(self.tesseract_dim):
                    ranks = [
                        h * self.tensor_parallel_size + i + self.tesseract_dim * (j + self.tesseract_dim * k)
                        for k in range(self.tesseract_dep)
                    ]
                    group = dist.new_group(ranks)
                    group_cpu = dist.new_group(ranks, backend='gloo') if dist.get_backend() != 'gloo' else group

                    if self.rank in ranks:
                        local_rank = ranks.index(self.rank)
                        group_world_size = len(ranks)
                        process_group = group
                        cpu_group = group_cpu
                        ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


# i row j col k dep
class Initializer_2p5D_XZ(ProcessGroupInitializer):
    """2.5d tensor parallel initialization among cols times dep.

    Args:
        tesseract_dim (int): The dimension of tesseract.
        tesseract_dep (int): The dimension of depth.
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """

    def __init__(self, tesseract_dim: int, tesseract_dep: int, *args):
        super(Initializer_2p5D_XZ, self).__init__(*args)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.tesseract_dep = tesseract_dep
        self.tesseract_dim = tesseract_dim

    def init_dist_group(self):
        """Initialize 2.5D tensor colXdepth parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                2.5D tensor colXdepth parallelism's information in a tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_2P5D_XZ

        for h in range(self.num_group):
            for i in range(self.tesseract_dim):
                ranks = [
                    h * self.tensor_parallel_size + i + self.tesseract_dim * (j + self.tesseract_dim * k)
                    for k in range(self.tesseract_dep)
                    for j in range(self.tesseract_dim)
                ]
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
class Initializer_2p5D(ProcessGroupInitializer):
    """
    Serve as the single entry point to Tesseract parallel initialization.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
        depth (int): The depth of 2.5d parallel.
    """

    def __init__(self, rank: int, world_size: int, config: Config, data_parallel_size: int, pipeline_parallel_size: int,
                 tensor_parallel_size: int, depth: int):
        args = (rank, world_size, config, data_parallel_size, pipeline_parallel_size, tensor_parallel_size)
        super().__init__(*args)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.tesseract_dim = int(math.sqrt(self.tensor_parallel_size / depth))
        self.tesseract_dep = depth

        assert self.tensor_parallel_size == self.tesseract_dim ** 2 * self.tesseract_dep, \
            "2.5D tesseract dim should equal to (tensor parallel size / tesseract dep) ^ 0.5"
        _check_tesseract_env_var(self.tesseract_dim, self.tesseract_dep)

        self.col_initializer = Initializer_2p5D_Col(self.tesseract_dim, self.tesseract_dep, *args)
        self.row_initializer = Initializer_2p5D_ROW(self.tesseract_dim, self.tesseract_dep, *args)
        self.dep_initializer = Initializer_2p5D_Dep(self.tesseract_dim, self.tesseract_dep, *args)
        self.xz_initializer = Initializer_2p5D_XZ(self.tesseract_dim, self.tesseract_dep, *args)

    def init_dist_group(self):
        """Initialize 2.5D tensor row, col, depth, and colXdepth parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            List[Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode)]:
                Whole 2.5D tensor parallelism's information in a list of tuples.
        """
        parallel_setting = [
            self.col_initializer.init_dist_group(),
            self.row_initializer.init_dist_group(),
            self.dep_initializer.init_dist_group(),
            self.xz_initializer.init_dist_group()
        ]
        return parallel_setting
