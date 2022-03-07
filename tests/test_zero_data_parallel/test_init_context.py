#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.zero.shard_utils.tensor_shard_strategy import TensorShardStrategy
from colossalai.zero.init_ctx import ZeroInitContext
from common import CONFIG, Net
from colossalai.utils import free_port


def run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    with ZeroInitContext(convert_fp16=True, convert_cuda=True, shard_strategy=TensorShardStrategy(), shard_param=True):
        # Note Net(checkpoint=True).cuda() moving to cuda is useless
        model = Net(checkpoint=True)

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
