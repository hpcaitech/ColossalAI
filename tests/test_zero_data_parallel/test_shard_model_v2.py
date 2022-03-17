#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
from asyncio.log import logger
from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.logging import get_dist_logger
from colossalai.utils import free_port
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import (BucketTensorShardStrategy, TensorShardStrategy)
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_model._zero3_utils import cast_tensor_to_fp16
from colossalai.zero.sharded_model.utils import col_model_deepcopy
from tests.components_to_test.registry import non_distributed_component_funcs
from torch.nn.parallel import DistributedDataParallel as DDP

from common import CONFIG, check_grads_padding, run_fwd_bwd


def run_dist(rank, world_size, port, use_zero_init_ctx, enable_autocast, shard_strategy):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    logger = get_dist_logger()
    logger.set_level('DEBUG')
    test_models = ['repeated_computed_layers', 'resnet18', 'bert']
    shard_strategy = shard_strategy()
    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, _, _, criterion = get_components_func()

        rm_torch_payload_on_the_fly = False

        if use_zero_init_ctx:
            with ZeroInitContext(convert_fp16=True,
                                 target_device=torch.device(f'cpu:0'),
                                 shard_strategy=shard_strategy,
                                 shard_param=True,
                                 rm_torch_payload_on_the_fly=rm_torch_payload_on_the_fly):
                zero_model = model_builder(checkpoint=True)
            zero_model = ShardedModelV2(zero_model, shard_strategy, use_memory_tracer=True)

            model = model_builder(checkpoint=True).half()
            col_model_deepcopy(zero_model, model)
            model = model.cuda()
        else:
            model = model_builder(checkpoint=True).half().cuda()
            zero_model = ShardedModelV2(copy.deepcopy(model), shard_strategy)

        model = DDP(model)

        for i, (data, label) in enumerate(train_dataloader):
            if i > 3:
                break

            data, label = cast_tensor_to_fp16(data).cuda(), label.cuda()
            run_fwd_bwd(model, data, label, criterion, enable_autocast)
            run_fwd_bwd(zero_model, data, label, criterion, enable_autocast)

            check_grads_padding(model, zero_model, loose=True)

        # logger.debug('overall cuda ', zero_model._memstats_collector._overall_cuda)
        # logger.debug('model cuda ', zero_model._memstats_collector._model_data_cuda)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("enable_autocast", [True])
@pytest.mark.parametrize("use_zero_init_ctx", [True])
@pytest.mark.parametrize("shard_strategy", [TensorShardStrategy, BucketTensorShardStrategy])
def test_shard_model_v2(world_size, use_zero_init_ctx, enable_autocast, shard_strategy):
    run_func = partial(run_dist,
                       world_size=world_size,
                       port=free_port(),
                       use_zero_init_ctx=use_zero_init_ctx,
                       enable_autocast=enable_autocast,
                       shard_strategy=shard_strategy)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_shard_model_v2(world_size=2, use_zero_init_ctx=True, enable_autocast=True, shard_strategy=TensorShardStrategy)
