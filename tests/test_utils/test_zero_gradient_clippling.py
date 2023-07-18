#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils import checkpoint, clip_grad_norm_fp32
from colossalai.zero.legacy.shard_utils.tensor_shard_strategy import TensorShardStrategy
from colossalai.zero.legacy.sharded_model.sharded_model_v2 import ShardedModelV2


def checkpoint_wrapper(module, enable=True):
    if enable:
        module.forward = partial(checkpoint, module.forward, False)
    return module


class Net(nn.Module):

    def __init__(self, checkpoint=False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)
        if checkpoint:
            self.fc1 = checkpoint_wrapper(self.fc1)
        self.layers = [self.fc1, self.fc2, self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def run_step(model, optimizer, x, enable_autocast=False, norm_type=2.0):
    model.train()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=enable_autocast):
        y = model(x)
        loss = y.sum()
    loss = loss.float()
    loss.backward()
    clip_grad(model, norm_type)
    optimizer.step()


def clip_grad(model, norm_type):
    if isinstance(model, DDP):
        clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=norm_type)
    else:
        clip_grad_norm_fp32(model.parameters(), max_norm=1.0, norm_type=norm_type)


def allclose(tensor_a: torch.Tensor, tensor_b: torch.Tensor, loose=False) -> bool:
    if loose:
        return torch.allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-3)
    return torch.allclose(tensor_a, tensor_b)


def check_grads(model, zero_model, loose=False):
    rank = dist.get_rank()
    for p, zero_p in zip(model.parameters(), zero_model.parameters()):
        zero_grad = zero_p.grad.clone().to(p.device)
        chunks = torch.flatten(p.grad).chunk(4)
        if rank >= len(chunks):
            continue
        grad = chunks[rank]
        if zero_p.zero_shard_padding > 0:
            zero_grad = zero_grad[:-zero_p.zero_shard_padding]
        assert grad.dtype == zero_grad.dtype
        assert allclose(grad, zero_grad, loose=loose)


def check_params(model, zero_model, loose=False):
    rank = dist.get_rank()
    for p, zero_p in zip(model.parameters(), zero_model.parameters()):
        zero_shard_padding = zero_p.zero_shard_padding
        zero_p = zero_p.clone().to(p.device)
        chunks = torch.flatten(p).chunk(4)
        if rank >= len(chunks):
            continue
        p = chunks[rank]
        if zero_shard_padding > 0:
            zero_p = zero_p[:-zero_shard_padding]
        assert p.dtype == zero_p.dtype
        assert allclose(p, zero_p, loose=loose)


def run_dist(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_zero_clip_grad():
    world_size = 4
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_zero_clip_grad()
