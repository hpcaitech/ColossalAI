#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from colossalai.context.parallel_mode import ParallelMode
from colossalai.context.random import add_seed, seed, set_mode
from colossalai.utils import checkpoint


def forward(x, weight):
    out = torch.matmul(x, weight)
    with seed(ParallelMode.DATA):
        out_ = F.dropout(out, p=0.4, training=True)
    return out_


@pytest.mark.gpu
def test_activation_checkpointing():
    add_seed(ParallelMode.GLOBAL, 1024)
    set_mode(ParallelMode.GLOBAL)
    global_cuda_rng_state = torch.cuda.get_rng_state()
    add_seed(ParallelMode.DATA, 1026)
    set_mode(ParallelMode.DATA)
    data_parallel_cuda_rng_state = torch.cuda.get_rng_state()
    set_mode(ParallelMode.GLOBAL)

    # normal
    data = torch.rand(2, 2, requires_grad=True).cuda()
    data.retain_grad()
    weight = torch.rand(2, 4, requires_grad=True).cuda()

    data_ = data.clone().detach()
    data_.requires_grad = True
    data_.retain_grad()
    weight_ = weight.clone().detach()
    weight_.requires_grad = True

    out = forward(data, weight)
    loss = out.sum()
    loss.backward()

    # checkpoint
    set_mode(ParallelMode.GLOBAL)
    torch.cuda.set_rng_state(global_cuda_rng_state)
    set_mode(ParallelMode.DATA)
    torch.cuda.set_rng_state(data_parallel_cuda_rng_state)
    set_mode(ParallelMode.GLOBAL)
    out = checkpoint(forward, data_, weight_)
    loss = out.sum()
    loss.backward()

    assert torch.all(data.grad == data_.grad), 'Gradient of the input does not match'


if __name__ == '__main__':
    test_activation_checkpointing()
