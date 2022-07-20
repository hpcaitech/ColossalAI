#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy
from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import (BucketTensorShardStrategy, TensorShardStrategy)
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_model.utils import col_model_deepcopy
from tests.components_to_test.registry import non_distributed_component_funcs

from common import CONFIG


@parameterize("shard_strategy_class", [TensorShardStrategy, BucketTensorShardStrategy])
def run_zero_state_dict(shard_strategy_class):
    test_models = ['repeated_computed_layers', 'resnet18']
    shard_strategy = shard_strategy_class()
    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, test_dataloader, optimizer, criterion = get_components_func()

        with ZeroInitContext(target_device=torch.device('cuda', torch.cuda.current_device()),
                             shard_strategy=shard_strategy,
                             shard_param=True):
            zero_model = model_builder(checkpoint=True)
        zero_model = ShardedModelV2(zero_model, shard_strategy)

        model = model_builder(checkpoint=True).half()
        col_model_deepcopy(zero_model, model)
        model = model.cuda()

        zero_state_dict = zero_model.state_dict()
        for key, val in model.state_dict().items():
            assert torch.equal(val, zero_state_dict[key].to(val.device))


def run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_zero_state_dict()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
@rerun_if_address_is_in_use()
def test_zero_state_dict(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_zero_state_dict(2)
