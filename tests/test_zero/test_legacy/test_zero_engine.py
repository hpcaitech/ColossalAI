#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import torch
import torch.distributed as dist
from common import MP_PARALLEL_CONFIG, ZERO_PARALLEL_CONFIG, check_params, check_sharded_model_params
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.core import global_context as gpc
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.zero.legacy.init_ctx import ZeroInitContext
from colossalai.zero.legacy.sharded_model.utils import col_model_deepcopy
from colossalai.zero.low_level._utils import has_inf_or_nan
from tests.components_to_test.registry import non_distributed_component_funcs


def run_dist(rank, world_size, port, parallel_config, bf16):
    is_mp_config = parallel_config == MP_PARALLEL_CONFIG
    is_zero_config = parallel_config == ZERO_PARALLEL_CONFIG
    if bf16:
        parallel_config['zero']['model_config']['bf16'] = True
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
        with ZeroInitContext(target_device=torch.cuda.current_device(),
                             shard_strategy=gpc.config.zero.model_config.shard_strategy,
                             shard_param=True,
                             bf16=bf16):
            colo_model = model_builder(checkpoint=True)

        colo_optimizer = optimizer_class(colo_model.parameters(), lr=1e-3)
        engine, train_dataloader, _, _ = colossalai.initialize(colo_model,
                                                               optimizer=colo_optimizer,
                                                               criterion=criterion,
                                                               train_dataloader=train_dataloader)
        dtype = torch.bfloat16 if bf16 else torch.float16
        torch_model = model_builder(checkpoint=True).to(dtype)
        col_model_deepcopy(engine.model, torch_model)
        torch_model = torch_model.cuda().float()

        engine.train()
        torch_optimizer = optimizer_class(torch_model.parameters(), lr=1e-3)

        if dist.get_world_size() > 1:
            torch_model = DDP(torch_model, device_ids=[torch.cuda.current_device()])

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

            for param in torch_model.parameters():
                if param.grad is not None:
                    assert not has_inf_or_nan(param.grad)

            torch_optimizer.step()
            i += 1

        if is_mp_config:
            check_params(torch_model, colo_model, loose=True)
        elif is_zero_config:
            check_sharded_model_params(torch_model, colo_model, loose=True)


# FIXME: enable this test in next PR
@pytest.mark.skip
@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2, 4])
@rerun_if_address_is_in_use()
def test_mp_engine(world_size):
    spawn(run_dist, world_size, parallel_config=MP_PARALLEL_CONFIG)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("bf16", [True, False])
@rerun_if_address_is_in_use()
def test_zero_engine(world_size, bf16):
    spawn(run_dist, world_size, parallel_config=ZERO_PARALLEL_CONFIG, bf16=bf16)


if __name__ == '__main__':
    test_zero_engine(world_size=4)
