#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import torch
import torch.multiprocessing as mp

from colossalai.core import global_context as gpc
from colossalai.initialize import launch, get_default_parser
from functools import partial
from checks_1d.check_layer_1d import *

CONFIG = dict(
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(
            size=2,
            mode='1d'
        )
    ),
)


def check_layer(rank, world_size):
    launch(config=CONFIG,
           rank=rank,
           world_size=world_size,
           host='localhost',
           port=29920,
           backend='nccl')

    check_linear_col()
    check_linear_row()
    check_attention()
    check_mlp()
    check_patch_embedding()
    check_embed()
    check_head()

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_1d():
    world_size = 2
    run_func = partial(check_layer, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_1d()
