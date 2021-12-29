#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from colossalai import launch
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import free_port

CONFIG_PATH = Path(__file__).parent.joinpath('configs/parallel_2d_init.py').absolute()


def check_data_parallel_rank(rank):
    if rank in [0, 1, 2, 3, 4, 5, 6, 7]:
        assert gpc.get_local_rank(ParallelMode.DATA) == 0
    elif rank in [8, 9, 10, 11, 12, 13, 14, 15]:
        assert gpc.get_local_rank(ParallelMode.DATA) == 1


def check_pipeline_parallel_rank(rank):
    if rank in [0, 1, 2, 3]:
        assert gpc.get_local_rank(ParallelMode.PIPELINE) == 0
    elif rank in [4, 5, 6, 7]:
        assert gpc.get_local_rank(ParallelMode.PIPELINE) == 1
    elif rank in [8, 9, 10, 11]:
        assert gpc.get_local_rank(ParallelMode.PIPELINE) == 0
    elif rank in [12, 13, 14, 15]:
        assert gpc.get_local_rank(ParallelMode.PIPELINE) == 1


def check_tensor_parallel_rank(rank):
    if rank in [0, 4, 8, 12]:
        assert gpc.get_local_rank(ParallelMode.TENSOR) == 0
    elif rank in [1, 5, 9, 13]:
        assert gpc.get_local_rank(ParallelMode.TENSOR) == 1
    elif rank in [2, 6, 10, 14]:
        assert gpc.get_local_rank(ParallelMode.TENSOR) == 2
    elif rank in [3, 7, 11, 15]:
        assert gpc.get_local_rank(ParallelMode.TENSOR) == 3


def check_2d_parallel_rank(rank):
    if rank in [0, 4, 8, 12]:
        assert gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL) == 0
        assert gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW) == 0
    elif rank in [1, 5, 9, 13]:
        assert gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL) == 0
        assert gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW) == 1
    elif rank in [2, 6, 10, 14]:
        assert gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL) == 1
        assert gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW) == 0
    elif rank in [3, 7, 11, 15]:
        assert gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL) == 1
        assert gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW) == 1


def init_2d(rank, world_size, backend, port, host):
    dist_args = dict(
        config=CONFIG_PATH,
        rank=rank,
        world_size=world_size,
        backend=backend,
        port=port,
        host=host,
        verbose=True
    )
    launch(**dist_args)

    check_tensor_parallel_rank(rank)
    check_data_parallel_rank(rank)
    check_2d_parallel_rank(rank)
    check_pipeline_parallel_rank(rank)
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.cpu
def test_2d_init():
    """
    As no computation or communication is done, we can run this test on CPU.
    """
    world_size = 16
    test_fn = partial(init_2d,
                      world_size=world_size,
                      backend='gloo',
                      port=free_port(),
                      host='localhost'
                      )
    mp.spawn(test_fn, nprocs=world_size)


if __name__ == '__main__':
    test_2d_init()
