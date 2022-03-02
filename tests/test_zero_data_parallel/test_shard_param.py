#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.zero.shard_param import ShardParam
from colossalai.utils import free_port
from colossalai.logging import get_dist_logger, disable_existing_loggers
from tests.test_zero_data_parallel.common import Net, CONFIG


def run_init_shard_param(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    param = torch.nn.Parameter(data=torch.rand(2, 3))
    sparam = ShardParam(param, None, True)
    payload = sparam.payload(torch.device('cuda'))
    assert (list(payload.shape) == [3])


def run_shard_param_check(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    logger = get_dist_logger()
    model = Net()

    # add an attribute as ca_attr to hijack the access to param.data
    for _, param in model.named_parameters():
        numel_ref = (param.numel() + world_size - 1) // world_size
        param.ca_attr = ShardParam(param)
        param.ca_attr.shard()
        param_data = param.ca_attr.payload(torch.device('cpu'))
        assert (numel_ref == param_data.numel())

    for _, param in model.named_parameters():
        param.ca_attr.gather()
        param_data = param.ca_attr.payload(torch.device('cpu'))

    disable_existing_loggers([logger])


@pytest.mark.dist
def test_shard_shape():
    world_size = 2
    run_func = partial(run_shard_param_check, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


@pytest.mark.dist
def test_init_shard_param():
    world_size = 2
    run_func = partial(run_init_shard_param, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_shard_shape()
    test_init_shard_param()
