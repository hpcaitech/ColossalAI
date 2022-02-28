#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from asyncio.log import logger
from functools import partial

import colossalai
import pytest
import torch
from torch import nn
import torch.multiprocessing as mp
from colossalai.zero.shard_param import ShardParam
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from colossalai.utils import free_port
from colossalai.logging import get_dist_logger, disable_existing_loggers


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

class Net(nn.Module):
    def __init__(self, checkpoint=False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 5)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def run_shard_param_check(rank, world_size, port):
    colossalai.launch(config=CONFIG,
                      rank=rank,
                      world_size=world_size,
                      host='localhost',
                      port=port,
                      backend='nccl')
    
    logger = get_dist_logger()
    model = Net()

    # add an attribute as ca_attr to hijack the access to param.data
    for _, param in model.named_parameters():
        numel_ref = (param.numel() + world_size - 1) // world_size
        param.ca_attr = ShardParam(param)
        param.ca_attr.shard()
        param_data = param.ca_attr.payload(torch.device('cpu'))
        logger.info(f'shard {param_data.shape} {param_data}', ranks = [1])
        assert(numel_ref == param_data.numel())

    for _, param in model.named_parameters():
        param.ca_attr.gather()
        param_data = param.ca_attr.payload(torch.device('cpu'))
        logger.info(f'gather {param_data.shape} {param_data}', ranks = [1])
    
    disable_existing_loggers([logger])

@pytest.mark.dist
def test_run_shard_shape():
    world_size = 2
    run_func = partial(run_shard_param_check, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_run_shard_shape()