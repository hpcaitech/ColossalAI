#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import torch
from common import CONFIG, check_grads_padding, run_fwd_bwd
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.zero.legacy.init_ctx import ZeroInitContext
from colossalai.zero.legacy.shard_utils import BucketTensorShardStrategy
from colossalai.zero.legacy.sharded_model import ShardedModelV2
from colossalai.zero.legacy.sharded_model._utils import cast_tensor_to_fp16
from colossalai.zero.legacy.sharded_model.utils import col_model_deepcopy
from tests.components_to_test.registry import non_distributed_component_funcs


@parameterize("enable_autocast", [True])
@parameterize("shard_strategy_class", [BucketTensorShardStrategy])
def run_model_test(enable_autocast, shard_strategy_class):
    test_models = ['repeated_computed_layers', 'resnet18', 'bert', 'hanging_param_model']
    shard_strategy = shard_strategy_class()
    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, _, _, criterion = get_components_func()

        with ZeroInitContext(target_device=torch.device('cuda', torch.cuda.current_device()),
                             shard_strategy=shard_strategy,
                             shard_param=True):
            zero_model = model_builder(checkpoint=True)
        zero_model = ShardedModelV2(zero_model, shard_strategy)

        model = model_builder(checkpoint=True).half()
        col_model_deepcopy(zero_model, model)
        model = model.cuda()

        model = DDP(model, device_ids=[torch.cuda.current_device()])

        for i, (data, label) in enumerate(train_dataloader):
            if i > 5:
                break

            data, label = cast_tensor_to_fp16(data).cuda(), label.cuda()
            run_fwd_bwd(model, data, label, criterion, enable_autocast)
            run_fwd_bwd(zero_model, data, label, criterion, enable_autocast)

            check_grads_padding(model, zero_model, loose=True)


def run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_model_test()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
@rerun_if_address_is_in_use()
def test_shard_model_v2(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_shard_model_v2(world_size=2)
