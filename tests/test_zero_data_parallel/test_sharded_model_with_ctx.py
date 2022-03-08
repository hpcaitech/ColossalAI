#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
from functools import partial

import colossalai
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from colossalai.utils import free_port
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils.tensor_shard_strategy import \
    TensorShardStrategy
from colossalai.zero.sharded_model import ShardedModelV2
from tests.components_to_test.registry import non_distributed_component_funcs
from torch.nn.parallel import DistributedDataParallel as DDP

from common import CONFIG, check_grads, check_grads_padding


def run_fwd_bwd(model, data, label, criterion, enable_autocast=False):
    model.train()
    with torch.cuda.amp.autocast(enabled=enable_autocast):
        y = model(data)
        loss = criterion(y, label)
    loss = loss.float()
    if isinstance(model, ShardedModelV2):
        model.backward(loss)
    else:
        loss.backward()


def run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    test_models = ['repeated_computed_layers', 'resnet18']
    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        shard_strategy = TensorShardStrategy()
        with ZeroInitContext(convert_fp16=True, convert_cuda=True, shard_strategy=shard_strategy, shard_param=True):
            zero_model, train_dataloader, test_dataloader, optimizer, criterion = get_components_func()
            zero_model = zero_model()
        model = copy.deepcopy(zero_model)
        zero_model = ShardedModelV2(zero_model, shard_strategy)
        model_state_dict = zero_model.state_dict()
        for n, p in model.named_parameters():
            p.data = model_state_dict[n]
        model = model.half().cuda()
        if dist.get_world_size() > 1:
            model = DDP(model)

        for i, (data, label) in enumerate(train_dataloader):
            if i > 2:
                break
            data, label = data.half().cuda(), label.cuda()
            run_fwd_bwd(model, data, label, criterion, False)
            run_fwd_bwd(zero_model, data, label, criterion, False)
            if dist.get_world_size() > 1:
                check_grads_padding(model, zero_model, loose=True)
            else:
                check_grads(model, zero_model, loose=True)


@pytest.mark.dist
def test_shard_model_v2():
    world_size = 2
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_shard_model_v2()
