#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
from functools import partial

import colossalai
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import free_port
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
from torch.optim import Adam

from common import (CONFIG, Net, check_grads, check_grads_padding, check_params, check_sharded_params_padding)


def run_step(model, optimizer, x, enable_autocast=False):
    model.train()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=enable_autocast):
        y = model(x)
        loss = y.sum()
    loss = loss.float()
    if isinstance(model, ShardedModelV2):
        optimizer.backward(loss)
    else:
        loss.backward()
    optimizer.step()


def run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    model = Net(checkpoint=True).cuda()
    zero_model = copy.deepcopy(model)
    zero_model = ShardedModelV2(zero_model, process_group=gpc.get_group(ParallelMode.DATA))
    for n, p in zero_model.named_parameters():
        p._name = n
    optim = Adam(model.parameters(), lr=1e-3)
    sharded_optim = ShardedOptimizerV2(Adam(zero_model.parameters(), lr=1e-3), zero_model)

    for _ in range(2):
        x = torch.rand(2, 5).cuda()
        run_step(zero_model, sharded_optim, x, False)
        run_step(model, optim, x, False)
        if dist.get_world_size() > 1:
            check_grads_padding(model, zero_model)
            check_sharded_params_padding(model, zero_model)
        else:
            check_grads(model, zero_model)
            check_params(model, zero_model)


@pytest.mark.skip
def test_sharded_optim_v2():
    world_size = 2
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_sharded_optim_v2()
