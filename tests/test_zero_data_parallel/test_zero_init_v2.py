#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
from functools import partial
import pytest

import colossalai
from colossalai.utils import free_port

import torch
import torch.multiprocessing as mp

from tests.components_to_test.registry import non_distributed_component_funcs

from common import check_sharded_params_padding


def run_dist(rank, world_size, port):
    _config = dict(fp16=dict(mode=None,),
                   zero=dict(optimzer=dict(optimizer_type=torch.optim.Adam, optimizer_config=dict(lr=1e-3)),
                             offload_optimizer_config=dict(device='cpu',
                                                           pin_memory=True,
                                                           buffer_count=5,
                                                           fast_init=False),
                             offload_param_config=dict(device='cpu',
                                                       pin_memory=True,
                                                       buffer_count=5,
                                                       buffer_size=1e8,
                                                       max_in_cpu=1e9)),
                   parallel=dict(pipeline=dict(size=1), tensor=dict(size=1, mode=None)))

    colossalai.launch(config=_config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    # FIXME revert back
    # test_models = ['repeated_computed_layers', 'resnet18', 'bert']
    test_models = ['bert']
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

                torch_model(data, label)
                torch_loss = engine.criterion(output, label)
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
