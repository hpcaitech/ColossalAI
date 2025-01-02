import os
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.testing import assert_close

import colossalai
from colossalai.lazy import LazyInitContext
from colossalai.shardformer.layer import FusedLinear, FusedLinear1D_Col, FusedLinear1D_Row
from colossalai.shardformer.layer.qkv_fused_linear import split_fused_qkv_in_gpt2_style
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn

# This code is copied from https://github.com/huggingface/transformers
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"


@parameterize("lazy_init", [False, True])
def check_linear_1d_col(lazy_init: bool):
    ctx = LazyInitContext() if lazy_init else nullcontext()
    linear = nn.Linear(8, 80).cuda()
    with ctx:
        linear_copy = nn.Linear(8, 80).cuda()
    linear_col = FusedLinear1D_Col.from_native_module(
        linear_copy, process_group=None, gather_output=True, split_sizes=[32, 32, 16]
    )

    assert linear.weight.shape == torch.Size([80, 8])
    assert linear.bias.shape == torch.Size([80])
    assert linear_col.weight.shape == torch.Size([40, 8])
    assert linear_col.bias.shape == torch.Size([40])
    assert linear_copy.weight is linear_col.weight
    assert linear_copy.bias is linear_col.bias

    # ensure weights are reversibly loadable
    linear_col.load_state_dict(linear.state_dict())
    linear.load_state_dict(linear_col.state_dict())

    # check computation correctness
    x = torch.rand(4, 8).cuda()
    out = linear(x)
    gather_out = linear_col(x)
    assert_close(out, gather_out)

    # check backward correctness
    out.sum().backward()
    gather_out.sum().backward()

    target_grad = split_fused_qkv_in_gpt2_style(linear.weight.grad, [32, 32, 16], None, False)
    assert_close(target_grad, linear_col.weight.grad)


@parameterize("lazy_init", [False, True])
def check_linear_1d_row(lazy_init: bool):
    ctx = LazyInitContext() if lazy_init else nullcontext()

    linear = nn.Linear(80, 8).cuda()
    with ctx:
        linear_copy = nn.Linear(80, 8).cuda()
    linear_row = FusedLinear1D_Row.from_native_module(
        linear_copy, process_group=None, split_sizes=[32, 32, 16], parallel_input=False
    )

    assert linear.weight.shape == torch.Size([8, 80])
    assert linear_row.weight.shape == torch.Size([8, 40])
    assert linear_row.bias.shape == torch.Size([8])
    assert linear_copy.weight is linear_row.weight
    assert linear_copy.bias is linear_row.bias

    # ensure weights are reversibly loadable
    linear_row.load_state_dict(linear.state_dict())
    linear.load_state_dict(linear_row.state_dict())

    # check computation correctness
    x = torch.rand(4, 80).cuda()
    out = linear(x)
    gather_out = linear_row(x)
    assert_close(out, gather_out)

    # check backward correctness
    out.sum().backward()
    gather_out.sum().backward()

    target_grad = split_fused_qkv_in_gpt2_style(linear.weight.grad, [32, 32, 16], None, True)
    assert_close(target_grad, linear_row.weight.grad)


@parameterize("lazy_init", [False, True])
def check_linear_1d_col_row(lazy_init: bool):
    ctx = LazyInitContext() if lazy_init else nullcontext()

    linear1 = nn.Linear(8, 80).cuda()
    linear2 = nn.Linear(80, 8).cuda()
    with ctx:
        linear1_copy = nn.Linear(8, 80).cuda()
        linear2_copy = nn.Linear(80, 8).cuda()
    linear_col = FusedLinear1D_Col.from_native_module(linear1_copy, process_group=None, split_sizes=[32, 32, 16])
    linear_row = FusedLinear1D_Row.from_native_module(
        linear2_copy,
        process_group=None,
        split_sizes=[32, 32, 16],
    )
    # ensure weights are reversibly loadable
    linear_col.load_state_dict(linear1.state_dict())
    linear_row.load_state_dict(linear2.state_dict())

    # check computation correctness
    x = torch.rand(4, 8).cuda()
    target_out = linear2(linear1(x))
    out = linear_row(linear_col(x))
    assert_close(out, target_out)

    # check backward correctness
    target_out.sum().backward()
    out.sum().backward()

    target_grad1 = split_fused_qkv_in_gpt2_style(linear1.weight.grad, [32, 32, 16], None, False)
    assert_close(target_grad1, linear_col.weight.grad)
    target_grad2 = split_fused_qkv_in_gpt2_style(linear2.weight.grad, [32, 32, 16], None, True)
    assert_close(target_grad2, linear_row.weight.grad)


@parameterize("lazy_init", [False, True])
def check_linear_1d_base(lazy_init: bool):
    ctx = LazyInitContext() if lazy_init else nullcontext()
    linear = nn.Linear(8, 80).cuda()
    with ctx:
        linear_copy = nn.Linear(8, 80).cuda()
    linear_base = FusedLinear.from_native_module(linear_copy)

    assert linear.weight.shape == torch.Size([80, 8])
    assert linear.bias.shape == torch.Size([80])
    assert linear_base.weight.shape == torch.Size([80, 8])
    assert linear_base.bias.shape == torch.Size([80])
    assert linear_copy.weight is linear_base.weight
    assert linear_copy.bias is linear_base.bias

    # ensure weights are reversibly loadable
    linear_base.load_state_dict(linear.state_dict())
    linear.load_state_dict(linear_base.state_dict())

    # check computation correctness
    x = torch.rand(4, 8).cuda()
    out = linear(x)
    base_out = linear_base(x)
    assert_close(out, base_out)

    # check backward correctness
    out.sum().backward()
    base_out.sum().backward()

    assert_close(linear.weight.grad, linear_base.weight.grad)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    check_linear_1d_col()
    check_linear_1d_row()
    check_linear_1d_col_row()
    check_linear_1d_base()


@rerun_if_address_is_in_use()
def test_linearconv():
    spawn(run_dist, nprocs=2)


if __name__ == "__main__":
    test_linearconv()
