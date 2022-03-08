#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy
from functools import partial

import colossalai
from colossalai.zero.sharded_param.sharded_param import ShardedParamV2
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.zero.sharded_param import ShardedTensor, ShardedParam
from colossalai.utils import free_port
from colossalai.logging import get_dist_logger, disable_existing_loggers
from tests.test_zero_data_parallel.common import Net, CONFIG, allclose


def run_shard_tensor(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    t = ShardedTensor(tensor=torch.randn(world_size * 2, 3))
    assert list(t.origin_shape) == [world_size * 2, 3]
    assert list(t.shape) == [world_size * 2, 3]

    shard_strategy = TensorShardStrategy(process_group=None)

    # test shard strategy
    shard_strategy.shard([t])
    assert list(t.shape) == [6], f"{list(t.shape)} vs 6"
    shard_strategy.gather([t])
    assert list(t.shape) == [world_size * 2, 3], f"{list(t.shape)} vs {[world_size * 2, 3]}"


@pytest.mark.dist
def test_shard_tensor():
    world_size = 2
    run_func = partial(run_shard_tensor, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


def _run_shard_param_v2(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    param = torch.nn.Parameter(torch.randn(2, 3))
    param_ref = deepcopy(param)
    sparam = ShardedParamV2(param=param, process_group=None)

    allclose(sparam.data, param_ref.data)

    sparam.remove_torch_payload()
    assert (param.data.numel() == 1)


@pytest.mark.dist
def test_shard_param_v2():
    world_size = 2
    run_func = partial(_run_shard_param_v2, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


def _run_test_shard_param(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    param = torch.nn.Parameter(torch.randn(2, 3))
    param_ref = deepcopy(param)
    sparam = ShardedParamV2(param=param, process_group=None)
    print(sparam.data)
    print(param_ref.data)

    logger = get_dist_logger()
    model = Net()

    # add an attribute as ca_attr to hijack the access to param.data
    for _, param in model.named_parameters():
        numel_ref = (param.numel() + world_size - 1) // world_size
        param.ca_attr = ShardedParam(param)
        param.ca_attr.shard()
        param_data = param.ca_attr.payload(torch.device('cpu'))
        assert (numel_ref == param_data.numel())

    for _, param in model.named_parameters():
        param.ca_attr.gather()
        param_data = param.ca_attr.payload(torch.device('cpu'))

    disable_existing_loggers([logger])


@pytest.mark.dist
def test_shard_param():
    world_size = 2
    run_func = partial(_run_test_shard_param, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


def run_init_shard_param(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    param = torch.nn.Parameter(data=torch.rand(2, 3))
    sparam = ShardedParam(param, None, True)
    payload = sparam.payload(torch.device('cuda'))
    assert (list(payload.shape) == [3])
    del sparam

    param_shape = (2, 3)
    sparam = ShardedParam(param_shape, process_group=None, is_sharded=True, device=torch.device('cpu'))
    payload = sparam.payload(torch.device('cuda'))
    assert (list(payload.shape) == [3])

    param_shape = (2, 3)
    sparam = ShardedParam(param_shape, process_group=None, is_sharded=False, device=torch.device('cpu'))
    payload = sparam.payload(torch.device('cuda'))
    assert (list(payload.shape) == [2, 3])


@pytest.mark.dist
def test_init_shard_param():
    world_size = 2
    run_func = partial(run_init_shard_param, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_shard_tensor()
    test_shard_param()
    test_shard_param_v2()
    test_init_shard_param()
