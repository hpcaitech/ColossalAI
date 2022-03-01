#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers
from colossalai.utils import checkpoint, free_port
from colossalai.zero.sharded_model import ShardedModelV2

CONFIG = dict(
    fp16=dict(
        mode=None,
    ),
    zero=dict(
        level=3,
        verbose=False,
        offload_optimizer_config=dict(
            device='cpu',
            pin_memory=True,
            buffer_count=5,
            fast_init=False
        ),
        offload_param_config=dict(
            device='cpu',
            pin_memory=True,
            buffer_count=5,
            buffer_size=1e8,
            max_in_cpu=1e9
        )
    ),
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(size=1, mode=None)
    )
)


def checkpoint_wrapper(module, enable=True):
    if enable:
        module.forward = partial(checkpoint, module.forward)
    return module


class Net(nn.Module):
    def __init__(self, checkpoint=False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)
        if checkpoint:
            self.fc1 = checkpoint_wrapper(self.fc1)
        self.layers = [
            self.fc1,
            self.fc2,
            self.fc1,
            self.fc2,
            self.fc3
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def allclose(tensor_a: torch.Tensor, tensor_b: torch.Tensor, loose=False) -> bool:
    if loose:
        return torch.allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-3)
    return torch.allclose(tensor_a, tensor_b)


def run_fwd_bwd(model, x, enable_autocast=False):
    model.train()
    with torch.cuda.amp.autocast(enabled=enable_autocast):
        y = model(x)
        loss = y.sum()
    loss = loss.float()
    # loss.backward()
    model.backward(loss)


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

    for _ in range(5):
        x = torch.rand(2, 5).cuda()
        run_fwd_bwd(zero_model, x, False)


@pytest.mark.dist
def test_shard_model_v2():
    world_size = 1
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_shard_model_v2()
