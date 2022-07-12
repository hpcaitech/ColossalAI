#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F
from colossalai.context.parallel_mode import ParallelMode
from colossalai.context.random import add_seed, seed, set_mode, reset_seeds
from colossalai.utils import checkpoint


def forward(x, weight):
    out = torch.matmul(x, weight)
    with seed(ParallelMode.DATA):
        out_ = F.dropout(out, p=0.4, training=True)
    return out_


@pytest.mark.gpu
@pytest.mark.skip("set seed error")
@pytest.mark.parametrize("cpu_offload", [True, False])
def test_activation_checkpointing(cpu_offload):

    # We put initilization here to avoid change cuda rng state below
    inputs = torch.rand(2, 2, requires_grad=True, device='cuda')
    weight = torch.rand(2, 4, requires_grad=True, device='cuda')

    # Get a copy of input tensors
    inputs_ = torch.empty(2, 2, requires_grad=True, device='cuda')
    inputs_.data.copy_(inputs.data)
    weight_ = torch.empty(2, 4, requires_grad=True, device='cuda')
    weight_.data.copy_(weight.data)

    add_seed(ParallelMode.GLOBAL, 1024)
    add_seed(ParallelMode.DATA, 1026)
    set_mode(ParallelMode.GLOBAL)
    global_cuda_rng_state = torch.cuda.get_rng_state()
    set_mode(ParallelMode.DATA)
    data_parallel_cuda_rng_state = torch.cuda.get_rng_state()
    set_mode(ParallelMode.GLOBAL)

    out = forward(inputs, weight)
    loss = out.sum()
    loss.backward()

    # Recover cuda rng states
    set_mode(ParallelMode.GLOBAL)
    torch.cuda.set_rng_state(global_cuda_rng_state)
    set_mode(ParallelMode.DATA)
    torch.cuda.set_rng_state(data_parallel_cuda_rng_state)
    set_mode(ParallelMode.GLOBAL)

    out = checkpoint(forward, cpu_offload, inputs_, weight_)
    loss = out.sum()
    loss.backward()

    assert torch.all(inputs.grad == inputs_.grad), 'Gradient of the input does not match'
    torch.cuda.empty_cache()

    # as seed manager is singleton
    # if we don't reset seeds here,
    # other tests will fail if running together with this test
    # as other tests can't overwrite the seed set by this test
    reset_seeds()
