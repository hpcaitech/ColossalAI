#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from functools import partial
from pathlib import Path

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.zero.sharded_model.param_manager import Zero3ParameterManager
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from colossalai.utils import free_port

BATCH_SIZE = 16
IMG_SIZE = 224

CONFIG = dict(
    fp16=dict(
        mode=None,
    ),
    zero=dict(
        level=3,
        verbose=False,
        offload_optimizer_config=dict(
            device='cpu',
            pin_memory=True,
            buffer_count=5,
            fast_init=False
        ),
        offload_param_config=dict(
            device='cpu',
            pin_memory=True,
            buffer_count=5,
            buffer_size=1e8,
            max_in_cpu=1e9
        )
    ),
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(size=1, mode=None)
    )
)


def run_shard_shape_check(rank, world_size, port):
    colossalai.launch(config=CONFIG,
                      rank=rank,
                      world_size=world_size,
                      host='localhost',
                      port=port,
                      backend='nccl')
    
    model = torch.nn.Linear(2, 4 * world_size)
    gpc.init_parallel_groups()
    Zero3ParameterManager(module=model, process_group=gpc.get_group(ParallelMode.DATA), offload_config=CONFIG.get('offload_param_config'))

    assert(model.weight.numel() == 4 * 2)
    assert(model.bias.numel() == 4)


@pytest.mark.dist
def test_run_shard_shape():
    world_size = 2
    run_func = partial(run_shard_shape_check, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_run_shard_shape()
