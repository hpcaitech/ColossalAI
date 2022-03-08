#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.zero.shard_utils.tensor_shard_strategy import TensorShardStrategy
from colossalai.zero.init_ctx import ZeroInitContext
from common import CONFIG
from colossalai.utils import free_port
from tests.components_to_test.registry import non_distributed_component_funcs


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
                assert hasattr(param, 'ca_attr')
                assert param.ca_attr.data.dtype == torch.half
                assert param.ca_attr._data_sharded_tensor.is_sharded
                assert param.ca_attr.data.device.type == 'cuda'


@pytest.mark.dist
def test_zero_init_context():
    world_size = 2
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_zero_init_context()
