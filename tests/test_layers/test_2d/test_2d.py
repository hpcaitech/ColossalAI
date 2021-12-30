#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from colossalai.core import global_context as gpc
from colossalai.initialize import launch
from colossalai.utils import free_port

from checks_2d.check_layer_2d import *
from checks_2d.check_operation_2d import *

CONFIG = dict(
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(
            size=4,
            mode='2d'
        )
    ),
)


def check_operations():
    check_AB()
    check_ABT()
    check_ATB()


def check_layer():
    check_linear()
    check_layernorm()
    check_classifier()

def check_layer_and_operation(rank, world_size, port):
    launch(config=CONFIG,
           rank=rank,
           world_size=world_size,
           host='localhost',
           port=port,
           backend='nccl')

    # check_operations()
    check_layer()
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_2d():
    world_size = 4
    run_func = partial(check_layer_and_operation, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_2d()
