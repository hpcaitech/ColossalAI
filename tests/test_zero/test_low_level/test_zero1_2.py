import copy

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import colossalai
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from colossalai.zero import LowLevelZeroOptimizer


class MlpModel(nn.Module):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(123, 253)
        self.linear_drop = nn.Linear(253, 253)
        self.linear2 = nn.Linear(253, 512)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


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


def split_ddp_grad(grad, world_size):
    with torch.no_grad():
        grad = grad.clone().detach().flatten()
        padding_size = (world_size - grad.numel() % world_size) % world_size
        if padding_size > 0:
            grad = torch.nn.functional.pad(grad, [0, padding_size])
        splited_grad = grad.split(grad.numel() // world_size)
    return splited_grad


@parameterize("fp8_communication", [True, False])
def exam_zero_1_2(fp8_communication: bool):
    """
    In this test, we want to test whether zero stage 1 and 2
    deliver the same numerical results despite different communication
    pattern

    we use these prefixes to differentiate the zero stage
    oss: partition optimizer states
    pg: partition gradients and optimizer states

    """
    local_rank = torch.distributed.get_rank()
    seed_all(2001)

    # create model
    zero1_model = MlpModel().cuda()
    zero2_model = copy.deepcopy(zero1_model)

    # create optimizer
    zero1_optimizer = torch.optim.Adam(zero1_model.parameters(), lr=1)
    zero2_optimizer = torch.optim.Adam(zero2_model.parameters(), lr=1)
    zero1_optimizer = LowLevelZeroOptimizer(
        zero1_optimizer,
        overlap_communication=True,
        initial_scale=128,
        verbose=True,
        fp8_communication=fp8_communication,
    )
    zero2_optimizer = LowLevelZeroOptimizer(
        zero2_optimizer,
        overlap_communication=True,
        partition_grad=True,
        initial_scale=128,
        fp8_communication=fp8_communication,
    )
    # create data
    seed_all(2001 + local_rank)
    input_data = torch.randn(32, 123).cuda()

    zero1_output = zero1_model(input_data)
    zero2_output = zero2_model(input_data)
    assert torch.equal(zero1_output, zero2_output)

    # zero-dp backward
    zero1_optimizer.backward(zero1_output.mean().float())
    zero2_optimizer.backward(zero2_output.mean().float())

    # check grad
    for p1, p2 in zip(zero1_model.parameters(), zero2_model.parameters()):
        g1 = zero1_optimizer.get_param_grad(p1)
        g2 = zero2_optimizer.get_param_grad(p2)
        if g1 is None or g2 is None:
            assert g1 is None and g2 is None
            continue
        if fp8_communication:
            loose_close(g1, g2, dtype=torch.float16)
        else:
            assert torch.allclose(g1, g2)

    # step
    zero1_optimizer.step()
    zero2_optimizer.step()

    # check updated param
    for z1p, z2p in zip(zero1_model.parameters(), zero2_model.parameters()):
        if not fp8_communication:
            assert torch.allclose(z1p, z2p)


@parameterize("dtype", [torch.float16, torch.bfloat16])
@parameterize("master_weights", [True, False])
def exam_zero_1_torch_ddp(world_size, dtype: torch.dtype, master_weights: bool):
    """
    In this test, two pairs of model and optimizers are created.
    1. zero: use sharded optimizer and fp16 parameters
    2. torch: use torch DDP and fp32 parameters

    We feed these two sets of models with the same input and check if the
    differences in model output and updated parameters are within tolerance.
    """
    local_rank = torch.distributed.get_rank()
    seed_all(1453)

    # create models
    torch_model = MlpModel().cuda().to(dtype)
    zero_model = copy.deepcopy(torch_model).to(dtype)

    torch_model = DDP(torch_model.cuda(), static_graph=True).cuda()

    # create optimizer
    zero_optimizer = torch.optim.SGD(zero_model.parameters(), lr=1)

    # we only test stage 1 here
    # in `check_sharded_param_consistency.py`, we will test whether
    # level 1 and 2 will produce exactly the same results
    zero_optimizer = LowLevelZeroOptimizer(
        zero_optimizer,
        overlap_communication=True,
        initial_scale=1,
        reduce_bucket_size=1024 * 1024,
        master_weights=master_weights,
    )

    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1)

    seed_all(1453 + local_rank)

    for _ in range(2):
        # create
        input_data = torch.rand(32, 123).cuda().to(dtype)

        # zero-dp forward
        zero_output = zero_model(input_data)

        # torch-ddp forward
        torch_output = torch_model(input_data)
        loose_close(zero_output, torch_output, dtype=dtype)

        # zero-dp backward
        zero_optimizer.backward(zero_output.mean())

        # torch-ddp backward
        torch_output.mean().backward()

        # check grad
        for (n, p), z1p in zip(torch_model.named_parameters(), zero_model.parameters()):
            zero_grad = zero_optimizer.get_param_grad(z1p)
            if p.grad is None:
                assert zero_grad is None
                continue
            loose_close(p.grad, zero_grad, dtype=dtype)

        # zero-dp step
        zero_optimizer.step()

        # torch ddp step
        torch_optimizer.step()

        zero_optimizer._force_wait_all_gather()

        # check updated param
        for (n, p), z1p in zip(torch_model.named_parameters(), zero_model.parameters()):
            loose_close(p, z1p, dtype=dtype)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")

    exam_zero_1_torch_ddp(world_size=world_size)
    exam_zero_1_2()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_zero_1_2():
    spawn(run_dist, 2)


if __name__ == "__main__":
    test_zero_1_2()
