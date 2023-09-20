import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.nn.layer.parallel_sequence import RingAV, RingQK
from colossalai.testing import rerun_if_address_is_in_use, spawn

CONFIG = dict(parallel=dict(tensor=dict(size=4, mode="sequence")))


def check_ring_qk(rank, world_size):
    # params
    batch_size = 4
    num_heads = 4
    seq_length = 32
    attention_head_size = 32
    sub_seq_length = seq_length // world_size

    # create master tensors
    q = torch.rand(batch_size * num_heads, seq_length, attention_head_size).cuda()
    k = torch.rand(batch_size * num_heads, seq_length, attention_head_size).cuda()
    dist.broadcast(q, src=0, group=gpc.get_group(ParallelMode.SEQUENCE))
    dist.broadcast(k, src=0, group=gpc.get_group(ParallelMode.SEQUENCE))

    # create distributed tensors
    sub_q = q.clone()[:, rank * sub_seq_length : (rank + 1) * sub_seq_length].contiguous()
    sub_k = k.clone()[:, rank * sub_seq_length : (rank + 1) * sub_seq_length].contiguous()

    # set autograd attributes
    q.requires_grad = True
    k.requires_grad = True
    q.retain_grad()
    k.retain_grad()
    sub_q.requires_grad = True
    sub_k.requires_grad = True
    sub_q.retain_grad()
    sub_k.retain_grad()

    # compute master attention scores
    a = torch.matmul(q, k.transpose(2, 1))

    # compute distributed attention scores
    ring_qk = RingQK.apply
    sub_a = ring_qk(sub_q, sub_k, batch_size, num_heads, sub_seq_length)

    # check master and distributed attention scores
    sub_master_a = a[:, rank * sub_seq_length : (rank + 1) * sub_seq_length]
    assert torch.allclose(sub_a, sub_master_a, rtol=1e-5, atol=1e-2)

    # run master backward
    a.retain_grad()
    a.mean().backward()

    # run distributed backward
    partial_master_a_grad = a.grad[:, rank * sub_seq_length : (rank + 1) * sub_seq_length]
    torch.autograd.backward(sub_a, partial_master_a_grad)

    # check master and distributed grads
    partial_master_q_grad = q.grad[:, rank * sub_seq_length : (rank + 1) * sub_seq_length]
    assert torch.allclose(sub_q.grad, partial_master_q_grad, rtol=1e-5, atol=1e-2), "attention score cannot match"


def check_ring_av(rank, world_size):
    # params
    batch_size = 4
    num_heads = 4
    seq_length = 16
    attention_head_size = 32
    sub_seq_length = seq_length // world_size

    # create master tensors
    a = torch.rand(batch_size * num_heads, seq_length, seq_length).cuda()
    v = torch.rand(batch_size * num_heads, seq_length, attention_head_size).cuda()
    dist.broadcast(a, src=0, group=gpc.get_group(ParallelMode.SEQUENCE))
    dist.broadcast(v, src=0, group=gpc.get_group(ParallelMode.SEQUENCE))

    # create distributed tensors
    sub_a = a.clone()[:, rank * sub_seq_length : (rank + 1) * sub_seq_length].contiguous()
    sub_v = v.clone()[:, rank * sub_seq_length : (rank + 1) * sub_seq_length].contiguous()

    # set autograd attributes
    a.requires_grad = True
    v.requires_grad = True
    a.retain_grad()
    v.retain_grad()
    sub_a.requires_grad = True
    sub_v.requires_grad = True
    sub_a.retain_grad()
    sub_v.retain_grad()

    # compute master attention scores
    out = torch.matmul(a, v)

    # compute distributed attention scores
    ring_av = RingAV.apply
    sub_out = ring_av(sub_a, sub_v, batch_size, num_heads, attention_head_size, sub_seq_length)

    # print(f'master output shape: {out.shape}, partial output shape: {sub_out.shape}')

    # check master and distributed output
    sub_master_out = out[:, rank * sub_seq_length : (rank + 1) * sub_seq_length]
    assert torch.allclose(sub_out, sub_master_out, rtol=1e-5, atol=1e-2)

    # # run master backward
    out.retain_grad()
    out.mean().backward()

    # # run distributed backward
    partial_master_out_grad = out.grad[:, rank * sub_seq_length : (rank + 1) * sub_seq_length]
    torch.autograd.backward(sub_out, partial_master_out_grad)

    # # check master and distributed grads
    partial_master_a_grad = a.grad[:, rank * sub_seq_length : (rank + 1) * sub_seq_length]
    assert torch.allclose(sub_a.grad, partial_master_a_grad, rtol=1e-5, atol=1e-2), "attention output cannot match"


def run_test(rank, world_size, port):
    colossalai.legacy.launch(rank=rank, world_size=world_size, config=CONFIG, host="localhost", port=port)

    # check_ring_qk(rank, world_size)
    check_ring_av(rank, world_size)

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_sequence():
    spawn(run_test, 4)


if __name__ == "__main__":
    test_sequence()
