#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
from distutils.command.config import config
from functools import partial
from colossalai.zero.sharded_model.sharded_model_v2 import ShardedModelV2
from examples.language.bert.sequene_parallel import model
import pytest

import colossalai
from colossalai.utils import free_port

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tests.components_to_test.registry import non_distributed_component_funcs
from common import check_sharded_params_padding, ZERO_PARALLEL_CONFIG, MP_PARALLEL_CONFIG, check_params


def run_dist(rank, world_size, port, parallel_config):
    colossalai.launch(config=parallel_config,
                      rank=rank,
                      world_size=world_size,
                      host='localhost',
                      port=port,
                      backend='nccl')

    test_models = ['repeated_computed_layers', 'resnet18', 'bert']
    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, _, optimizer_class, criterion = get_components_func()

        colo_model = model_builder(checkpoint=True)
        torch_model = copy.deepcopy(colo_model).cuda()
        engine, train_dataloader, _, _ = colossalai.initialize(colo_model,
                                                               optimizer=optimizer_class,
                                                               criterion=criterion,
                                                               train_dataloader=train_dataloader)
        engine.train()
        torch_optimizer = optimizer_class(torch_model.parameters())

        if dist.get_world_size() > 1:
            torch_model = DDP(torch_model)

        i = 0
        for data, label in train_dataloader:
            if i > 4:
                break

            data, label = data.cuda(), label.cuda()

            engine.zero_grad()
            torch_optimizer.zero_grad()

            if criterion:
                output = engine(data)
                loss = engine.criterion(output, label)

                torch_output = torch_model(data)
                torch_loss = engine.criterion(torch_output, label)
            else:
                loss = engine(data, label)
                torch_loss = torch_model(data, label)

            engine.backward(loss)
            engine.step()

            torch_loss.backward()
            torch_optimizer.step()
            i += 1

        # for torch_param, zero_param in zip(torch_model.parameters(), colo_model.parameters()):
        #     assert torch.allclose(torch_param, zero_param), f"diff {torch_param - zero_param}"

        if parallel_config == MP_PARALLEL_CONFIG:
            check_params(torch_model, colo_model, loose=True)
        elif isinstance(colo_model, ShardedModelV2):
            check_sharded_params_padding(torch_model, colo_model, loose=True)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2, 4])
def test_mp_engine(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port(), parallel_config=MP_PARALLEL_CONFIG)
    mp.spawn(run_func, nprocs=world_size)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
def test_zero_engine(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port(), parallel_config=ZERO_PARALLEL_CONFIG)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_zero_engine(world_size=4)
