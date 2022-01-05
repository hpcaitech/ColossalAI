#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.initialize import launch
from colossalai.utils import free_port

CONFIG_PATH = Path(__file__).parent.joinpath('configs/parallel_3d_init.py').absolute()


def check_data_parallel_rank(rank):
    dp_rank = gpc.get_local_rank(ParallelMode.DATA)

    if rank in list(range(16)):
        assert dp_rank == 0
    elif rank in list(range(16, 32)):
        assert dp_rank == 1


def check_pipeline_parallel_rank(rank):
    ppr = gpc.get_local_rank(ParallelMode.PIPELINE)

    if rank in list(range(8)):
        assert ppr == 0
    elif rank in list(range(8, 16)):
        assert ppr == 1
    elif rank in list(range(16, 24)):
        assert ppr == 0
    elif rank in list(range(24, 32)):
        assert ppr == 1


def check_model_parallel_rank(rank):
    for i in range(16):
        if rank in [i, i+16]:
            assert gpc.get_local_rank(ParallelMode.MODEL) == i


def check_tensor_parallel_rank(rank):
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)

    for i in range(8):
        ranks = list(range(i, 32, 8))
        if rank in ranks:
            assert tp_rank == i


def check_3d_parallel_rank(rank):
    ip_rank = gpc.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    wp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    op_rank = gpc.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)

    # check for input parallel group
    for i in range(2):
        _ranks = list(range(i * 2, 32, 4))
        _ranks_plus_one = [val + 1 for val in _ranks]
        input_ranks = _ranks + _ranks_plus_one
        if rank in input_ranks:
            assert ip_rank == i

    # check for weight parallel group
    for i in range(2):
        ranks = list(range(i, 32, 2))

        if rank in ranks:
            assert wp_rank == i

    # check for output parallel group
    for i in range(2):
        ranks = []
        for j in range(i * 4, 32, 8):
            ranks.extend([j + k for k in range(4)])
        if rank in ranks:
            assert op_rank == i


def init_3d(rank, world_size, backend, port, host):
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
    check_3d_parallel_rank(rank)
    check_data_parallel_rank(rank)
    check_pipeline_parallel_rank(rank)
    check_model_parallel_rank(rank)
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.cpu
def test_3d_init():
    """
    As no computation or communication is done, we can run this test on CPU.
    """
    world_size = 32
    test_fn = partial(init_3d,
                      world_size=world_size,
                      backend='gloo',
                      port=free_port(),
                      host='localhost'
                      )
    mp.spawn(test_fn, nprocs=world_size)


if __name__ == '__main__':
    test_3d_init()
