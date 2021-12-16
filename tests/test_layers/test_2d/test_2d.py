#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import torch
import torch.multiprocessing as mp

from colossalai.core import global_context as gpc
from colossalai.initialize import launch, get_default_parser
from checks_2d.check_layer_2d import check_linear, check_layernorm, check_attention, check_mlp, check_transformerlayer
from checks_2d.check_operation_2d import check_AB, check_ABT, check_ATB
from functools import partial


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
    check_attention()
    check_mlp()
    check_transformerlayer()


def check_layer_and_operation(rank, world_size):
    launch(config=CONFIG,
           rank=rank,
           world_size=world_size,
           host='localhost',
           port=29921,
           backend='nccl')

    check_operations()
    check_layer()
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_2d():
    world_size = 4
    run_func = partial(check_layer_and_operation, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_2d()
