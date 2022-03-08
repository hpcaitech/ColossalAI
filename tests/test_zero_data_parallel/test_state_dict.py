#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy
from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.utils import free_port
from colossalai.zero.shard_utils.tensor_shard_strategy import \
    TensorShardStrategy
from colossalai.zero.sharded_model import ShardedModelV2
from tests.components_to_test.registry import non_distributed_component_funcs

from common import CONFIG


def run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    test_models = ['repeated_computed_layers', 'resnet18']
    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model, train_dataloader, test_dataloader, optimizer, criterion = get_components_func()
        model = model()
        shard_strategy = TensorShardStrategy()
        model = model.half().cuda()
        zero_model = ShardedModelV2(deepcopy(model), shard_strategy)
        zero_state_dict = zero_model.state_dict()
        for key, val in model.state_dict().items():
            assert torch.equal(val, zero_state_dict[key])


@pytest.mark.dist
def test_zero_state_dict():
    world_size = 2
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_zero_state_dict()
