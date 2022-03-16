#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
from functools import partial

import colossalai
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from colossalai.nn.optimizer import CPUAdam
from colossalai.utils import free_port
from colossalai.zero.shard_utils import (BucketTensorShardStrategy, TensorShardStrategy)
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
from tests.components_to_test.registry import non_distributed_component_funcs
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from colossalai.utils import get_current_device


def run_dist(rank, world_size, port):
    CONFIG = dict(fp16=dict(mode=None,),
                  zero=dict(offload_optimizer_config=dict(device='cpu',
                                                          pin_memory=True,
                                                          buffer_count=5,
                                                          fast_init=False),
                            offload_param_config=dict(device='cpu',
                                                      pin_memory=True,
                                                      buffer_count=5,
                                                      buffer_size=1e8,
                                                      max_in_cpu=1e9)),
                  parallel=dict(pipeline=dict(size=1), tensor=dict(size=1, mode=None)))

    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    test_models = ['repeated_computed_layers', 'resnet18', 'bert']
    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, _, optimizer_class, criterion = get_components_func()

        engine, train_dataloader, _, _ = colossalai.initialize(model=model_builder(checkpoint=True),
                                                               optimizer=optimizer_class,
                                                               criterion=criterion,
                                                               train_dataloader=train_dataloader)

        engine.train()
        i = 0
        for data, label in train_dataloader:
            if i > 5:
                break

            data, label = data.cuda(), label.cuda()

            engine.zero_grad()
            if criterion:
                output = engine(data)
                loss = engine.criterion(output, label)
            else:
                loss = engine(output, label)
            engine.backward(loss)
            engine.step()


@pytest.mark.dist
# @pytest.mark.parametrize("world_size", [1, 2])
# @pytest.mark.parametrize("shard_strategy", [TensorShardStrategy, BucketTensorShardStrategy])
def test_zero_init(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_zero_init(world_size=2)
