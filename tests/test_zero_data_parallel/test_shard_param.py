#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy
from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.utils import free_port
from colossalai.zero.shard_utils import (BucketTensorShardStrategy, TensorShardStrategy)
from colossalai.zero.sharded_param import ShardedParam, ShardedTensor
from colossalai.zero.sharded_param.sharded_param import ShardedParamV2
from tests.components_to_test.registry import non_distributed_component_funcs
from tests.test_zero_data_parallel.common import CONFIG, allclose


def _run_shard_tensor(rank, world_size, port, shard_strategy):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    t = ShardedTensor(tensor=torch.randn(world_size * 2, 3))
    assert list(t.origin_shape) == [world_size * 2, 3]
    assert list(t.shape) == [world_size * 2, 3]

    shard_strategy = shard_strategy(process_group=None)

    # test shard strategy
    shard_strategy.shard([t])
    assert list(t.shape) == [6], f"{list(t.shape)} vs 6"
    shard_strategy.gather([t])
    assert list(t.shape) == [world_size * 2, 3], f"{list(t.shape)} vs {[world_size * 2, 3]}"


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("shard_strategy", [TensorShardStrategy, BucketTensorShardStrategy])
def test_shard_tensor(world_size, shard_strategy):
    run_func = partial(_run_shard_tensor, world_size=world_size, port=free_port(), shard_strategy=shard_strategy)
    mp.spawn(run_func, nprocs=world_size)


def _run_shard_param_v2(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    param = torch.nn.Parameter(torch.randn(2, 3))
    param_ref = deepcopy(param)
    sparam = ShardedParamV2(param=param, process_group=None)

    allclose(sparam.data.payload, param_ref.data)

    sparam.remove_torch_payload()
    assert (param.data.numel() == 1)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
def test_shard_param_v2(world_size):
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
    for get_components_func in non_distributed_component_funcs:
        model_builder, *_ = get_components_func()
        model = model_builder(checkpoint=True)
        # add an attribute as col_attr to hijack the access to param.data
        for _, param in model.named_parameters():
            numel_ref = (param.numel() + world_size - 1) // world_size
            param.col_attr = ShardedParam(param)
            param.col_attr.shard()
            param_data = param.col_attr.payload(torch.device('cpu'))
            assert (numel_ref == param_data.numel())

        for _, param in model.named_parameters():
            param.col_attr.gather()
            param_data = param.col_attr.payload(torch.device('cpu'))

        disable_existing_loggers([logger])


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
def test_shard_param(world_size):
    run_func = partial(_run_test_shard_param, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


def _run_init_shard_param(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    param = torch.nn.Parameter(data=torch.rand(world_size, 3))
    sparam = ShardedParam(param, None, True)
    payload = sparam.payload(torch.device('cuda'))
    assert (list(payload.shape) == [3])
    del sparam

    param_shape = (world_size, 3)
    sparam = ShardedParam(param_shape, process_group=None, is_sharded=True, device=torch.device('cpu'))
    payload = sparam.payload(torch.device('cuda'))
    assert (list(payload.shape) == [3])

    param_shape = (world_size, 3)
    sparam = ShardedParam(param_shape, process_group=None, is_sharded=False, device=torch.device('cpu'))
    payload = sparam.payload(torch.device('cuda'))
    assert (list(payload.shape) == [world_size, 3])


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
def test_init_shard_param(world_size):
    run_func = partial(_run_init_shard_param, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_shard_tensor(2, TensorShardStrategy)
    test_shard_param(2)
    test_shard_param_v2(2)
    test_init_shard_param(4)
