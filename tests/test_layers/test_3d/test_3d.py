#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from colossalai.initialize import get_default_parser, launch

from checks_3d.check_layer_3d import *
from checks_3d.check_operation_3d import *

CONFIG = dict(
    parallel=dict(
        pipeline=1,
        tensor=dict(mode='3d', size=8),
    ),
    seed=42,
)

# def check_operations():
#     check_AB()
#     check_ABT()
#     check_ATB()
#     check_add()
#     check_mul()
#     check_sum()


def check_layer():
    check_linear()
    check_layernorm()
    check_attention()
    check_mlp()
    check_head()
    check_embed()
    check_loss()

           
def check_layer_and_operation(rank, world_size):
    launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=29923, backend='nccl')
    check_layer()
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_3d():
    world_size = 8
    run_func = partial(check_layer_and_operation, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)

    torch.cuda.synchronize()


if __name__ == '__main__':
    test_3d()
