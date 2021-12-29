#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from colossalai.core import global_context as gpc
from colossalai.initialize import launch
from colossalai.utils import free_port

from checks_1d.check_layer_1d import *

CONFIG = dict(
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(
            size=4,
            mode='1d'
        )
    ),
)


def check_layer(rank, world_size, port):
    launch(config=CONFIG,
           rank=rank,
           world_size=world_size,
           host='localhost',
           port=port,
           backend='nccl')

    check_linear_col()
    check_linear_row()

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_1d():
    world_size = 4
    run_func = partial(check_layer, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_1d()
