#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.utils import free_port
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils.tensor_shard_strategy import \
    TensorShardStrategy
from tests.components_to_test.registry import non_distributed_component_funcs

from common import CONFIG
from colossalai.utils.memory_tracer.allocator import GLOBAL_MODEL_DATA_TRACER


def run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    for get_components_func in non_distributed_component_funcs:
        model_builder, _, _, _, _ = get_components_func()
        with ZeroInitContext(convert_fp16=True,
                             convert_cuda=True,
                             shard_strategy=TensorShardStrategy(),
                             shard_param=True):
            model = model_builder(checkpoint=True)

        for param in model.parameters():
            assert hasattr(param, 'col_attr')
            assert param.col_attr.data.dtype == torch.half
            assert param.col_attr.data.is_sharded
            assert param.col_attr.data.payload.device.type == 'cuda'

    print(f'cuda usgae {GLOBAL_MODEL_DATA_TRACER.cuda_usage}')
    assert (GLOBAL_MODEL_DATA_TRACER.cuda_usage > 0)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
def test_zero_init_context(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_zero_init_context(2)
