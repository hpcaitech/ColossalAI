#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial
import pytest

import torch
import torch.multiprocessing as mp

import colossalai
from colossalai.zero.sharded_model.param_manager import Zero3ParameterManager
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from colossalai.utils import free_port
from common import CONFIG


def run_shard_shape_check(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    model = torch.nn.Linear(2, 4 * world_size)
    gpc.init_parallel_groups()
    Zero3ParameterManager(module=model,
                          process_group=gpc.get_group(ParallelMode.DATA),
                          offload_config=CONFIG.get('offload_param_config'))

    assert (model.weight.numel() == 4 * 2)
    assert (model.bias.numel() == 4)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_run_shard_shape(world_size):
    run_func = partial(run_shard_shape_check, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_run_shard_shape(2)
