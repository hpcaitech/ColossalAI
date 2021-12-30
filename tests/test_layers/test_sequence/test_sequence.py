#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import torch
import torch.multiprocessing as mp
from colossalai.initialize import launch
from colossalai.logging import get_dist_logger
from checks_seq.check_layer_seq import *
from functools import partial
from colossalai.utils import free_port


CONFIG = dict(
    parallel=dict(
        pipeline=1,
        tensor=dict(mode='sequence', size=4)
    )
)


def check_layer():
    check_selfattention()


def run_check_sequence(rank, world_size, port):
    # init dist
    launch(config=CONFIG,
           rank=rank,
           world_size=world_size,
           host='localhost',
           port=port,
           backend='nccl')
    logger = get_dist_logger()
    logger.info('Distributed environment is initialzied.', ranks=[0])

    # check layers
    check_layer()
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_sequence():
    world_size = 4
    run_func = partial(run_check_sequence, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_sequence()
