#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
from functools import partial
from operator import mod
from pyexpat import model

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.logging import disable_existing_loggers
from colossalai.utils import free_port
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from tests.test_zero_data_parallel.common import Net, CONFIG, check_grads


def run_fwd_bwd(model, x, enable_autocast=False):
    model.train()
    with torch.cuda.amp.autocast(enabled=enable_autocast):
        y = model(x)
        loss = y.sum()
    loss = loss.float()
    loss.backward()
    

def run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG,
                      rank=rank,
                      world_size=world_size,
                      host='localhost',
                      port=port,
                      backend='nccl')
    
    model = Net(checkpoint=True).cuda()
    zero_model = copy.deepcopy(model)
    zero_model = ShardedModelV2(zero_model, process_group=gpc.get_group(ParallelMode.DATA))

    for _ in range(2):
        x = torch.rand(2, 5).cuda()
        run_fwd_bwd(zero_model, x, False)
        run_fwd_bwd(model, x, False)
        check_grads(model, zero_model)


@pytest.mark.dist
def test_shard_model_v2():
    world_size = 2
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_shard_model_v2()
