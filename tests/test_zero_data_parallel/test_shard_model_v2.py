#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
from functools import partial
import pytest

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.utils import free_port
from colossalai.zero.shard_utils.tensor_shard_strategy import \
    TensorShardStrategy
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_model._zero3_utils import cast_tensor_to_fp16

from tests.components_to_test.registry import non_distributed_component_funcs
from common import CONFIG, check_grads_padding


def run_fwd_bwd(model, data, label, criterion, enable_autocast=False):
    model.train()
    with torch.cuda.amp.autocast(enabled=enable_autocast):
        if criterion:
            y = model(data)
            loss = criterion(y, label)
        else:
            loss = model(data, label)
        loss = loss.float()
    if isinstance(model, ShardedModelV2):
        model.backward(loss)
    else:
        loss.backward()


def run_dist(rank, world_size, port, enable_autocast):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    test_models = ['repeated_computed_layers', 'resnet18', 'bert']
    shard_strategy = TensorShardStrategy()
    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model, train_dataloader, _, _, criterion = get_components_func()
        model = model(checkpoint=True).half().cuda()
        zero_model = ShardedModelV2(copy.deepcopy(model), shard_strategy)

        model = DDP(model)

        for i, (data, label) in enumerate(train_dataloader):
            if i > 3:
                break

            data, label = cast_tensor_to_fp16(data).cuda(), label.cuda()
            run_fwd_bwd(model, data, label, criterion, enable_autocast)
            run_fwd_bwd(zero_model, data, label, criterion, enable_autocast)

            check_grads_padding(model, zero_model, loose=True)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("enable_autocast", [True])
def test_shard_model_v2(world_size, enable_autocast):
    run_func = partial(run_dist, world_size=world_size, port=free_port(), enable_autocast=enable_autocast)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_shard_model_v2(world_size=2)
