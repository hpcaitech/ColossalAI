#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.utils import free_port
from colossalai.utils.cuda import get_current_device
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import (BucketTensorShardStrategy, TensorShardStrategy)
from tests.components_to_test.registry import non_distributed_component_funcs

from common import CONFIG
from colossalai.utils.memory_tracer.model_data_memtracer import GLOBAL_MODEL_DATA_TRACER


def run_dist(rank, world_size, port, init_device, shard_strategy):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    for get_components_func in non_distributed_component_funcs:
        model_builder, _, _, _, _ = get_components_func()
        model_numel_tensor = torch.zeros(1, dtype=torch.int)
        with ZeroInitContext(convert_fp16=True,
                             target_device=init_device,
                             shard_strategy=shard_strategy(),
                             shard_param=True,
                             model_numel_tensor=model_numel_tensor):
            model = model_builder(checkpoint=True)

        for param in model.parameters():
            assert hasattr(param, 'col_attr')
            assert param.col_attr.data.dtype == torch.half
            assert param.col_attr.data.is_sharded
            assert param.col_attr.data.payload.device.type == init_device.type, \
                f'{param.col_attr.data.payload.device.type} vs. {init_device.type}'

    print(f'cuda usgae {GLOBAL_MODEL_DATA_TRACER.cuda_usage}')
    print(f'numel {model_numel_tensor}')
    if init_device.type == 'cuda':
        assert (GLOBAL_MODEL_DATA_TRACER.cuda_usage > 0)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
@pytest.mark.parametrize("init_device", [torch.device('cpu'), torch.device(f'cuda:{get_current_device()}')])
@pytest.mark.parametrize("shard_strategy", [TensorShardStrategy, BucketTensorShardStrategy])
def test_zero_init_context(world_size, init_device, shard_strategy):
    run_func = partial(run_dist,
                       world_size=world_size,
                       port=free_port(),
                       init_device=init_device,
                       shard_strategy=shard_strategy)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    # test_zero_init_context(2, torch.device('cpu'), TensorShardStrategy)
    test_zero_init_context(4, torch.device('cpu'), BucketTensorShardStrategy)
