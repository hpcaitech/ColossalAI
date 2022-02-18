import os
from functools import partial
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import free_port, get_current_device
from colossalai.nn.layer.moe import Top2Router, MoeLayer
from colossalai.global_variables import moe_env


BATCH_SIZE = 32
NUM_EXPERTS = 4
CONFIG = dict(parallel=dict(moe=dict(size=4)))


def check_equal(A, B, atol=1e-06):
    assert torch.allclose(A, B, rtol=0, atol=atol) is True


def run_routing(rank, world_size, port, rs=2, hidden_size=128, data_type=torch.float32):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    # torch.set_printoptions(precision=30)
    torch.backends.cuda.matmul.allow_tf32 = False
    local_rank = gpc.get_local_rank(ParallelMode.GLOBAL)
    torch.manual_seed(rs + local_rank)
    moe_env.reset_loss()
    tokens = torch.randn(BATCH_SIZE, hidden_size,
                         dtype=data_type, device=get_current_device(), requires_grad=True)
    # print(f"tokens:\n{tokens}")
    router = Top2Router(1)
    layer = MoeLayer(hidden_size, NUM_EXPERTS, router, nn.Identity())
    if data_type == torch.float16:
        layer = layer.half()
    layer.cuda_mode = False

    old_out = layer(tokens)
    # print(f"old output:\n{old_out}")

    ech = old_out.shape
    grad = torch.randn(ech, device=get_current_device())
    old_out.backward(grad)

    o_tk_grad = tokens.grad.data.clone()
    o_gt_grad = layer.gate.weight.grad.data.clone()

    tokens.grad.zero_()
    layer.gate.weight.grad.zero_()

    layer.cuda_mode = True
    new_out = layer(tokens)

    # print(torch.max(torch.abs(old_out - new_out)))
    if data_type == torch.float32:
        check_equal(old_out, new_out)
    else:
        check_equal(old_out, new_out, 1e-2)
    # print(f"forward functions passed")

    # print(f"new output:\n{new_out}")
    new_out.backward(grad)
    n_tk_grad = tokens.grad.data.clone()
    n_gt_grad = layer.gate.weight.grad.data.clone()

    # print(torch.max(torch.abs(o_tk_grad - n_tk_grad)))
    if data_type == torch.float32:
        check_equal(o_tk_grad, n_tk_grad)
    else:
        check_equal(o_tk_grad, o_tk_grad, 1e-2)
    # print(f"tokens gradient passed")

    # print(torch.max(torch.abs(o_gt_grad - n_gt_grad)))
    if data_type == torch.float32:
        check_equal(o_gt_grad, n_gt_grad, 5e-05)
    else:
        check_equal(o_gt_grad, n_gt_grad, 2e-01)
    # print(f"linear weight gradient passed")


@pytest.mark.dist
@pytest.mark.parametrize("rs", [131])
@pytest.mark.parametrize("hidden_size", [32, 144])
@pytest.mark.parametrize("data_type", [torch.float32, torch.float16])
def test_moe_top2(rs, hidden_size, data_type):
    world_size = 4
    run_func = partial(run_routing, world_size=world_size, port=free_port(),
                       rs=rs, hidden_size=hidden_size, data_type=data_type)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_moe_top2(2, 256, torch.float16)
