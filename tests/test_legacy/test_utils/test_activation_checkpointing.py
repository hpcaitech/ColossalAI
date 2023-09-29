#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F

from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.context.random import add_seed, reset_seeds, seed, set_mode
from colossalai.legacy.utils.activation_checkpoint import checkpoint
from colossalai.testing import clear_cache_before_run, parameterize


def forward(x, weight):
    out = torch.matmul(x, weight)
    with seed(ParallelMode.DATA):
        out_ = F.dropout(out, p=0.4, training=True)
    return out_


def forward_inplace_ckpt(x, weight, cpu_offload=False):
    out = torch.matmul(x, weight)
    bn = torch.nn.BatchNorm1d(4, affine=False)
    bn = bn.to(device="cuda")
    out = bn(out)

    def ckpt0(x):
        return F.relu(x, inplace=True)

    out = checkpoint(ckpt0, cpu_offload, out, use_reentrant=False)
    return out


def forward_inplace(x, weight):
    out = torch.matmul(x, weight)
    bn = torch.nn.BatchNorm1d(4, affine=False)
    bn = bn.to(device="cuda")
    out = bn(out)
    out = F.relu(out, inplace=True)
    return out


@clear_cache_before_run()
@parameterize("use_reentrant", [True, False])
@parameterize("cpu_offload", [True, False])
def test_activation_checkpointing(cpu_offload, use_reentrant):
    # as seed manager is singleton
    # if we don't reset seeds here,
    # other tests might affect this test
    reset_seeds()

    # We put initialization here to avoid change cuda rng state below
    inputs = torch.rand(2, 2, requires_grad=True, device="cuda")
    weight = torch.rand(2, 4, requires_grad=True, device="cuda")

    # Get a copy of input tensors
    inputs_ = torch.empty(2, 2, requires_grad=True, device="cuda")
    inputs_.data.copy_(inputs.data)
    weight_ = torch.empty(2, 4, requires_grad=True, device="cuda")
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

    out = checkpoint(forward, cpu_offload, inputs_, weight_, use_reentrant=use_reentrant)
    loss = out.sum()
    loss.backward()

    assert torch.all(inputs.grad == inputs_.grad), "Gradient of the input does not match"
    torch.cuda.empty_cache()

    # Extra test for use_reentrant=False
    if use_reentrant == False:
        # Recover cuda rng states
        set_mode(ParallelMode.GLOBAL)
        torch.cuda.set_rng_state(global_cuda_rng_state)
        set_mode(ParallelMode.DATA)
        torch.cuda.set_rng_state(data_parallel_cuda_rng_state)
        set_mode(ParallelMode.GLOBAL)

        out = forward_inplace(inputs, weight)
        loss = out.sum()
        loss.backward()

        # Recover cuda rng states
        set_mode(ParallelMode.GLOBAL)
        torch.cuda.set_rng_state(global_cuda_rng_state)
        set_mode(ParallelMode.DATA)
        torch.cuda.set_rng_state(data_parallel_cuda_rng_state)
        set_mode(ParallelMode.GLOBAL)

        out = forward_inplace_ckpt(inputs_, weight_, cpu_offload=cpu_offload)
        loss = out.sum()
        loss.backward()

        assert torch.all(inputs.grad == inputs_.grad), "Gradient of the input does not match"
        torch.cuda.empty_cache()

    # as seed manager is singleton
    # if we don't reset seeds here,
    # other tests will fail if running together with this test
    # as other tests can't overwrite the seed set by this test
    reset_seeds()


if __name__ == "__main__":
    test_activation_checkpointing(False, False)
