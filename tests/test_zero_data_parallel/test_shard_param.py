#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from asyncio.log import logger
from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.zero.shard_param import ShardParam
from colossalai.utils import free_port
from colossalai.logging import get_dist_logger, disable_existing_loggers
from tests.test_zero_data_parallel.common import Net, CONFIG

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