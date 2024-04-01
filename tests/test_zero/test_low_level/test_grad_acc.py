import copy

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.testing import parameterize, spawn
from colossalai.testing.random import seed_all
from colossalai.utils import conditional_context
from colossalai.zero import LowLevelZeroOptimizer


def loose_close(a, b, dtype: torch.dtype = torch.float32):
    rtol = None
    atol = None
    if dtype is torch.float16:
        rtol = 5e-2
        atol = 5e-4
    elif dtype is torch.bfloat16:
        rtol = 4e-3
        atol = 4e-3

    a = a.detach().to(dtype)
    b = b.detach().to(dtype)

    assert_close(a, b, rtol=rtol, atol=atol)


class MlpModel(nn.Module):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 512)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


@parameterize("sub_dp_size", [1, 2])
def exam_zero_1_2_grad_acc(sub_dp_size: int):
    assert torch.distributed.get_world_size() % sub_dp_size == 0
    local_rank = torch.distributed.get_rank()
    seed_all(2009)
    device = get_accelerator().get_current_device()
    # create model
    zero1_model = MlpModel().to(device)
    zero2_model = copy.deepcopy(zero1_model)
    # create optimizer
    zero1_optimizer = torch.optim.Adam(zero1_model.parameters(), lr=1)
    zero2_optimizer = torch.optim.Adam(zero2_model.parameters(), lr=1)
    zero1_optimizer = LowLevelZeroOptimizer(
        zero1_optimizer,
        overlap_communication=True,
        initial_scale=32,
        clip_grad_norm=1.0,
        verbose=True,
        sub_dp_size=sub_dp_size,
    )
    zero2_optimizer = LowLevelZeroOptimizer(
        zero2_optimizer,
        overlap_communication=True,
        partition_grad=True,
        initial_scale=32,
        clip_grad_norm=1.0,
        sub_dp_size=sub_dp_size,
    )
    # create data
    seed_all(2021 + local_rank)
    input_data1 = torch.randn(32, 128, device=device)
    input_data2 = torch.randn(32, 128, device=device)

    def fwd_bwd_func(number, cur_data, check_flag):
        # zero-dp forward
        zero1_output = zero1_model(cur_data)
        zero2_output = zero2_model(cur_data)
        loose_close(zero1_output, zero2_output)

        # zero-dp backward
        zero1_optimizer.backward(zero1_output.sum().float())
        zero2_optimizer.backward(zero2_output.sum().float())

    fwd_bwd_func(0, input_data1, True)
    fwd_bwd_func(1, input_data2, False)

    # step
    zero1_optimizer.step()
    zero2_optimizer.step()

    # check updated param
    for z1p, z2p in zip(zero1_model.parameters(), zero2_model.parameters()):
        loose_close(z1p.data, z2p.data)


@parameterize("no_sync", [True, False])
@parameterize("sub_dp_size", [1, 2])
def exam_zero_1_grad_acc(no_sync: bool, sub_dp_size: int):
    assert torch.distributed.get_world_size() % sub_dp_size == 0
    local_rank = torch.distributed.get_rank()
    seed_all(2008)
    device = get_accelerator().get_current_device()

    # create models
    zero_model = MlpModel()
    torch_model = copy.deepcopy(zero_model)

    seed_all(2008)
    zero_model = zero_model.to(device)
    torch_model = DDP(torch_model.to(device), bucket_cap_mb=0)

    # create optimizer
    zero_optimizer = torch.optim.Adam(zero_model.parameters(), lr=1)

    # we only test stage 1 here
    # in `check_sharded_param_consistency.py`, we will test whether
    # level 1 and 2 will produce exactly the same results
    zero_optimizer = LowLevelZeroOptimizer(
        zero_optimizer,
        overlap_communication=False,
        reduce_bucket_size=262144,
        clip_grad_norm=1.0,
        sub_dp_size=sub_dp_size,
    )

    torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=1)

    # create data
    seed_all(2022 + local_rank)
    input_data1 = torch.randn(32, 128, device=device)
    input_data2 = torch.randn(32, 128, device=device)

    def fwd_bwd_func(no_sync, cur_data, check_flag):
        # zero1 fwd and bwd
        with conditional_context(zero_optimizer.no_sync(), no_sync):
            zero_output = zero_model(cur_data)
            zero_optimizer.backward(zero_output.sum().float())

        # torch-ddp fwd and bwd
        with conditional_context(torch_model.no_sync(), no_sync):
            torch_output = torch_model(cur_data)
            loose_close(zero_output, torch_output)
            torch_output.sum().backward()

        if check_flag:
            # check grad
            for (n, p), z1p in zip(torch_model.named_parameters(), zero_model.parameters()):
                loose_close(p.grad, z1p.grad)

    fwd_bwd_func(no_sync, input_data1, no_sync)
    fwd_bwd_func(False, input_data2, False)
    torch.nn.utils.clip_grad_norm_(torch_model.parameters(), 1.0)

    zero_optimizer.step()
    torch_optimizer.step()

    # check updated param
    for (n, p), z1p in zip(torch_model.named_parameters(), zero_model.parameters()):
        # print(n, p.shape, torch.max(p.data), torch.max(z1p.data), torch.max(torch.abs(p.data - z1p.data)))
        loose_close(p.data, z1p.data)


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host="localhost")

    exam_zero_1_grad_acc()
    exam_zero_1_2_grad_acc()


@pytest.mark.dist
def test_grad_accumulation():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_grad_accumulation()
