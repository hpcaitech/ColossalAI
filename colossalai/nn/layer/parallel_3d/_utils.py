#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os

from colossalai.constants import (DEPTH_3D, INPUT_GROUP_3D, OUTPUT_GROUP_3D,
                                  WEIGHT_GROUP_3D)
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from torch import Tensor


def get_depth_from_env() -> int:
    try:
        depth = os.environ[DEPTH_3D]
        depth = int(depth)
        assert depth > 0, 'DEPTH must be greater than zero'
        return depth

    except KeyError as e:
        raise EnvironmentError(
            'DEPTH is not found in the current environment, '
            'please make sure that you have used the correct process group initializer'
        )


def get_parallel_mode_from_env(group):
    return getattr(ParallelMode, os.environ[group])


def get_last_group(a, b):
    mapping = {
        ParallelMode.PARALLEL_3D_INPUT: 'A',
        ParallelMode.PARALLEL_3D_WEIGHT: 'B',
        ParallelMode.PARALLEL_3D_OUTPUT: 'C',
    }

    res = chr(
        ord('A') + ord('B') + ord('C') - ord(mapping[a]) - ord(mapping[b]))

    if res == 'A':
        return ParallelMode.PARALLEL_3D_INPUT
    elif res == 'B':
        return ParallelMode.PARALLEL_3D_WEIGHT
    elif res == 'C':
        return ParallelMode.PARALLEL_3D_OUTPUT


def swap_in_out_group():
    os.environ[INPUT_GROUP_3D], os.environ[OUTPUT_GROUP_3D] = \
        os.environ[OUTPUT_GROUP_3D], os.environ[INPUT_GROUP_3D]


def dbg_check_shape(tensor: Tensor, shape: tuple):
    rank = gpc.get_global_rank()
    if rank == 0:
        print(tensor.shape)
    assert tensor.shape == shape, \
        '{} does not match {}'.format(tensor.shape, shape)
