#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial
from pathlib import Path

import pytest
import torch.multiprocessing as mp

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.initialize import launch

CONFIG_PATH = Path(__file__).parent.joinpath('configs/parallel_2p5d_init.py').absolute()


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


def check_tensor_parallel_rank(rank):
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)

    for i in range(8):
        ranks = list(range(i, 32, 8))
        if rank in ranks:
            assert tp_rank == i, f'{rank}:{tp_rank}'


def check_2p5d_parallel_rank(rank):
    rp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
    cp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
    dp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)
    xp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_XZ)

    # check for row parallel group
    for i in range(2):
        ranks = list(range(i, 32, 2))
        if rank in ranks:
            assert rp_rank == i

    # check for col parallel group
    for i in range(2):
        ranks = list(range(i * 2, 32, 4))
        ranks_plus_ones = [val + 1 for val in ranks]
        ranks.extend(ranks_plus_ones)
        if rank in ranks:
            assert cp_rank == i

    # check for depth parallel group
    for i in range(2):
        ranks = []
        for j in range(i * 4, 32, 8):
            ranks.extend([j + k for k in range(4)])
        if rank in ranks:
            assert dp_rank == i

    # check for xz parallel group
    for i in range(2):
        ranks = list(range(i * 2, 32, 8))
        ranks_plus_one = [val + 1 for val in ranks]
        ranks.extend(ranks_plus_one)
        if rank in ranks:
            assert xp_rank == i


def init_2halfd(rank, world_size, backend, port, host):
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
    check_data_parallel_rank(rank)
    check_pipeline_parallel_rank(rank)
    check_tensor_parallel_rank(rank)
    check_2p5d_parallel_rank(rank)
    gpc.destroy()


@pytest.mark.cpu
def test_2halfd_init():
    """
    As no computation or communication is done, we can run this test on CPU.
    """
    world_size = 32
    test_fn = partial(init_2halfd,
                      world_size=world_size,
                      backend='gloo',
                      port='29501',
                      host='localhost'
                      )
    mp.spawn(test_fn, nprocs=world_size)


if __name__ == '__main__':
    test_2halfd_init()
