#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
from functools import partial
import pytest

import colossalai
from colossalai.utils import free_port

import torch.multiprocessing as mp

from tests.components_to_test.registry import non_distributed_component_funcs
from common import check_sharded_params_padding, ZERO_PARALLEL_CONFIG


def run_dist(rank, world_size, port):
    colossalai.launch(config=ZERO_PARALLEL_CONFIG,
                      rank=rank,
                      world_size=world_size,
                      host='localhost',
                      port=port,
                      backend='nccl')

    test_models = ['repeated_computed_layers', 'resnet18', 'bert']
    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, _, optimizer_class, criterion = get_components_func()

        # adapt to a Callbale with empty parameters
        # def module_builder_new():
        #     return model_builder(checkpoint=True)

        zero_model = model_builder(checkpoint=True)
        torch_model = copy.deepcopy(zero_model).cuda()
        engine, train_dataloader, _, _ = colossalai.initialize(zero_model,
                                                               optimizer=optimizer_class,
                                                               criterion=criterion,
                                                               train_dataloader=train_dataloader)
        engine.train()
        torch_optimizer = optimizer_class(torch_model.parameters())

        i = 0
        for data, label in train_dataloader:
            if i > 3:
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

        check_sharded_params_padding(torch_model, zero_model, loose=True)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
def test_zero_init(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_zero_init(world_size=2)
