import copy

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import colossalai
from colossalai.testing import spawn
from colossalai.testing.random import seed_all
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

    # create model
    zero1_model = MlpModel().cuda()
    zero2_model = copy.deepcopy(zero1_model)
    # create optimizer
    zero1_optimizer = torch.optim.Adam(zero1_model.parameters(), lr=1)
    zero2_optimizer = torch.optim.Adam(zero2_model.parameters(), lr=1)
    zero1_optimizer = LowLevelZeroOptimizer(zero1_optimizer,
                                            overlap_communication=True,
                                            initial_scale=32,
                                            clip_grad_norm=1.0,
                                            verbose=True)
    zero2_optimizer = LowLevelZeroOptimizer(zero2_optimizer,
                                            overlap_communication=True,
                                            partition_grad=True,
                                            initial_scale=32,
                                            clip_grad_norm=1.0)
    # create data
    seed_all(2021 + local_rank)
    input_data1 = torch.randn(32, 128).cuda()
    input_data2 = torch.randn(32, 128).cuda()

    def fwd_bwd_func(number, cur_data):
        # zero-dp forward
        zero1_output = zero1_model(cur_data)
        zero2_output = zero2_model(cur_data)
        assert torch.equal(zero1_output, zero2_output)

        # zero-dp backward
        zero1_optimizer.backward(zero1_output.sum().float(), sync_grad=False)
        zero2_optimizer.backward(zero2_output.sum().float(), sync_grad=False)

        for (n, z1p), z2p in zip(zero1_model.named_parameters(), zero2_model.parameters()):
            if z2p.grad is not None:
                # print(local_rank, n, z1p.shape, torch.max(z2p.grad), torch.max(torch.abs(z1p.grad - z2p.grad)))
                assert torch.equal(z1p.grad, z2p.grad)

        zero1_optimizer._sync_grad()
        zero2_optimizer._sync_grad()

    fwd_bwd_func(0, input_data1)
    fwd_bwd_func(1, input_data2)

    # step
    zero1_optimizer.step()
    zero2_optimizer.step()

    # check updated param
    for z1p, z2p in zip(zero1_model.parameters(), zero2_model.parameters()):
        assert torch.equal(z1p.data, z2p.data)


def exam_zero_1_grad_acc():
    local_rank = torch.distributed.get_rank()
    grad_scale = 32
    seed_all(2008)

    # create models
    zero_model = MlpModel()
    torch_model = copy.deepcopy(zero_model)

    seed_all(2008)
    zero_model = zero_model.cuda()
    torch_model = DDP(torch_model.cuda(), bucket_cap_mb=0)

    # create optimizer
    zero_optimizer = torch.optim.Adam(zero_model.parameters(), lr=1)

    # we only test stage 1 here
    # in `check_sharded_param_consistency.py`, we will test whether
    # level 1 and 2 will produce exactly the same results
    zero_optimizer = LowLevelZeroOptimizer(zero_optimizer,
                                           overlap_communication=False,
                                           initial_scale=grad_scale,
                                           reduce_bucket_size=262144,
                                           clip_grad_norm=1.0)

    torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=1)

    # create data
    seed_all(2022 + local_rank)
    input_data1 = torch.randn(32, 128).cuda()
    input_data2 = torch.randn(32, 128).cuda()

    def fwd_bwd_func(number, cur_data, check_flag):
        # zero-dp forward
        zero_output = zero_model(cur_data)

        # torch-ddp forward
        torch_output = torch_model(cur_data)
        assert torch.equal(zero_output, torch_output)

        # zero-dp backward
        zero_optimizer.backward(zero_output.sum().float(), sync_grad=False)
        # torch-ddp backward
        torch_output.sum().backward()

        if check_flag:
            # check grad
            for (n, p), z1p in zip(torch_model.named_parameters(), zero_model.parameters()):
                unscale_grad = z1p.grad / grad_scale
                # print(n, p.shape, torch.max(torch.abs(p.grad - unscale_grad)))
                assert torch.equal(p.grad, unscale_grad)

        zero_optimizer._sync_grad()

    fwd_bwd_func(0, input_data1, True)
    fwd_bwd_func(1, input_data2, False)

    zero_optimizer.step()
    torch.nn.utils.clip_grad_norm_(torch_model.parameters(), 1.0)
    torch_optimizer.step()

    # check updated param
    for (n, p), z1p in zip(torch_model.named_parameters(), zero_model.parameters()):
        # print(n, p.shape, torch.max(p.data), torch.max(z1p.data), torch.max(torch.abs(p.data - z1p.data)))
        assert_close(p.data, z1p.data)


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')

    exam_zero_1_grad_acc()
    exam_zero_1_2_grad_acc()


@pytest.mark.dist
def test_grad_accumulation():
    spawn(run_dist, 2)


if __name__ == '__main__':
    test_grad_accumulation()
