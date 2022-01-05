#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from colossalai.core import global_context as gpc
from colossalai.initialize import launch
from colossalai.utils import free_port

from checks_3d.check_layer_3d import *

CONFIG = dict(
    parallel=dict(
        pipeline=1,
        tensor=dict(mode='3d', size=8),
    ),
    seed=42,
)


def check_layer():
    check_linear()
    check_layernorm()
    check_classifier()
    # check_embed()
    # check_loss()


def check_layer_and_operation(rank, world_size, port):
    launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    check_layer()
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_3d():
    world_size = 8
    run_func = partial(check_layer_and_operation, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_3d()
