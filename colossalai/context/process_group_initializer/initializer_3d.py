#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import os

import torch.distributed as dist
from colossalai.constants import DEPTH_3D
from colossalai.registry import DIST_GROUP_INITIALIZER

from ..parallel_mode import ParallelMode
from .process_group_initializer import ProcessGroupInitializer


def _check_depth_env_var(depth):
    # check environment variable for SUMMA
    env_depth = os.environ.get(DEPTH_3D, None)

    if env_depth:
        assert int(env_depth) == depth, \
            'SUMMA_DIM has been set in the current environment and ' \
            'does not match with the value passed to this initialized'
    else:
        os.environ[DEPTH_3D] = str(depth)


class Initializer_3D_Input(ProcessGroupInitializer):
    '''2D tensor parallel initialization among input. 
    '''
    def __init__(self, num_group: int, depth: int, *args):
        super().__init__(*args)
        self.num_group = num_group
        self.depth = depth

    def init_dist_group(self):
        '''Initialize 3D tensor parallel groups among input, and assign local_ranks and groups to each gpu.

        :return: 3D tensor parallelism's information among input 
        :rtype: tuple(local_rank, group_world_size, process_group, ranks_in_group, mode)
        '''
        local_rank = None
        ranks_in_group = None
        process_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_3D_INPUT

        for h in range(self.num_group):
            for i in range(self.depth):
                for k in range(self.depth):
                    ranks = [
                        h * self.depth**3 + i + self.depth *
                        (j + self.depth * k) for j in range(self.depth)
                    ]
                    group = dist.new_group(ranks)

                    if self.rank in ranks:
                        local_rank = ranks.index(self.rank)
                        group_world_size = len(ranks)
                        process_group = group
                        ranks_in_group = ranks

        return local_rank, group_world_size, process_group, ranks_in_group, mode


class Initializer_3D_Weight(ProcessGroupInitializer):
    '''3D tensor parallel initialization among weight. 
    '''

    def __init__(self, num_group: int, depth: int, *args):
        super().__init__(*args)
        self.num_group = num_group
        self.depth = depth

    def init_dist_group(self):
        '''Initialize 3D tensor parallel groups among weight, and assign local_ranks and groups to each gpu.

        :return: 3D tensor parallelism's information among weight 
        :rtype: tuple(local_rank, group_world_size, process_group, ranks_in_group, mode)
        '''
        local_rank = None
        ranks_in_group = None
        process_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_3D_WEIGHT

        for h in range(self.num_group):
            for k in range(self.depth):
                for j in range(self.depth):
                    ranks = [
                        h * self.depth**3 + i + self.depth *
                        (j + self.depth * k) for i in range(self.depth)
                    ]
                    group = dist.new_group(ranks)

                    if self.rank in ranks:
                        local_rank = ranks.index(self.rank)
                        group_world_size = len(ranks)
                        process_group = group
                        ranks_in_group = ranks

        return local_rank, group_world_size, process_group, ranks_in_group, mode


class Initializer_3D_Output(ProcessGroupInitializer):
    '''2D tensor parallel initialization among weight. 
    '''

    def __init__(self, num_group: int, depth: int, *args):
        super().__init__(*args)
        self.num_group = num_group
        self.depth = depth

    def init_dist_group(self):
        '''Initialize 3D tensor parallel groups among output, and assign local_ranks and groups to each gpu.

        :return: 3D tensor parallelism's information among output 
        :rtype: tuple(local_rank, group_world_size, process_group, ranks_in_group, mode)
        '''
        local_rank = None
        ranks_in_group = None
        process_group = None
        group_world_size = None
        mode = ParallelMode.PARALLEL_3D_OUTPUT

        for h in range(self.num_group):
            for i in range(self.depth):
                for j in range(self.depth):
                    ranks = [
                        h * self.depth**3 + i + self.depth *
                        (j + self.depth * k) for k in range(self.depth)
                    ]
                    group = dist.new_group(ranks)

                    if self.rank in ranks:
                        local_rank = ranks.index(self.rank)
                        group_world_size = len(ranks)
                        process_group = group
                        ranks_in_group = ranks

        return local_rank, group_world_size, process_group, ranks_in_group, mode


@DIST_GROUP_INITIALIZER.register_module
class Initializer_3D(ProcessGroupInitializer):
    '''Serve as the single entry point to 3D parallel initialization.
    '''
    def __init__(self, *args):
        super().__init__(*args)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.depth = round(math.pow(self.tensor_parallel_size, 1 / 3))
        assert self.tensor_parallel_size == self.depth ** 3, \
            f'3D depth ({self.depth}) if not cube root of tensor parallel size ({self.tensor_parallel_size})'
        _check_depth_env_var(self.depth)

        self.input_initializer = Initializer_3D_Input(self.num_group,
                                                      self.depth, *args)
        self.weight_initializer = Initializer_3D_Weight(
            self.num_group, self.depth, *args)
        self.output_initializer = Initializer_3D_Output(
            self.num_group, self.depth, *args)

    def init_dist_group(self):
        '''Initialize 3D tensor parallel groups, and assign local_ranks and groups to each gpu.

        :return: 3D tensor parallelism's information 
        :rtype: list of tuples (local_rank, group_world_size, process_group, ranks_in_group, mode)
        '''
        parallel_setting = []
        parallel_setting.append(self.input_initializer.init_dist_group())
        parallel_setting.append(self.weight_initializer.init_dist_group())
        parallel_setting.append(self.output_initializer.init_dist_group())
        return parallel_setting
