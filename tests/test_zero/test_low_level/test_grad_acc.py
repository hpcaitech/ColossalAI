import copy

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.testing import spawn
from colossalai.testing.random import seed_all
from colossalai.utils import conditional_context
from colossalai.zero import LowLevelZeroOptimizer


class MlpModel(nn.Module):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 512)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def exam_zero_1_2_grad_acc():
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
        zero1_optimizer, overlap_communication=True, initial_scale=32, clip_grad_norm=1.0, verbose=True
    )
    zero2_optimizer = LowLevelZeroOptimizer(
        zero2_optimizer, overlap_communication=True, partition_grad=True, initial_scale=32, clip_grad_norm=1.0
    )
    # create data
    seed_all(2021 + local_rank)
    input_data1 = torch.randn(32, 128, device=device)
    input_data2 = torch.randn(32, 128, device=device)

    def fwd_bwd_func(number, cur_data, check_flag):
        # zero-dp forward
        zero1_output = zero1_model(cur_data)
        zero2_output = zero2_model(cur_data)
        assert torch.equal(zero1_output, zero2_output)

        # zero-dp backward
        zero1_optimizer.backward(zero1_output.sum().float())
        zero2_optimizer.backward(zero2_output.sum().float())

    fwd_bwd_func(0, input_data1, True)
    fwd_bwd_func(1, input_data2, False)

    # step
    zero1_optimizer.step()
    zero2_optimizer.step()

    zero1_optimizer._force_wait_all_gather()
    zero2_optimizer._force_wait_all_gather()

    # check updated param
    for z1p, z2p in zip(zero1_model.parameters(), zero2_model.parameters()):
        assert not hasattr(z1p, "_all_gather_handle")
        assert torch.equal(z1p.data, z2p.data)


def exam_zero_1_grad_acc(sync):
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
        zero_optimizer, overlap_communication=False, reduce_bucket_size=262144, clip_grad_norm=1.0
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
            assert torch.equal(zero_output, torch_output)
            torch_output.sum().backward()

        if check_flag:
            # check grad
            for (n, p), z1p in zip(torch_model.named_parameters(), zero_model.parameters()):
                assert torch.equal(p.grad, z1p.grad)

    fwd_bwd_func(sync, input_data1, sync)
    fwd_bwd_func(False, input_data2, False)

    zero_optimizer.step()
    torch.nn.utils.clip_grad_norm_(torch_model.parameters(), 1.0)
    torch_optimizer.step()

    # check updated param
    for (n, p), z1p in zip(torch_model.named_parameters(), zero_model.parameters()):
        # print(n, p.shape, torch.max(p.data), torch.max(z1p.data), torch.max(torch.abs(p.data - z1p.data)))
        assert_close(p.data, z1p.data)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")

    exam_zero_1_grad_acc(sync=True)
    exam_zero_1_grad_acc(sync=False)
    exam_zero_1_2_grad_acc()


@pytest.mark.dist
def test_grad_accumulation():
    spawn(run_dist, 2)


if __name__ == "__main__":
    test_grad_accumulation()
