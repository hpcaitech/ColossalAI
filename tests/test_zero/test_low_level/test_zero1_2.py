import copy

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import colossalai
from colossalai.testing import rerun_if_address_is_in_use, spawn
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


def half_close(a, b, loose=False):
    rtol = None
    atol = None
    if loose:
        rtol = 5e-2
        atol = 5e-4

    a = a.detach().half()
    b = b.detach().half()

    assert_close(a, b, rtol=rtol, atol=atol)


def exam_zero_1_2():
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
    zero1_optimizer = LowLevelZeroOptimizer(zero1_optimizer,
                                            overlap_communication=True,
                                            initial_scale=128,
                                            verbose=True)
    zero2_optimizer = LowLevelZeroOptimizer(zero2_optimizer,
                                            overlap_communication=True,
                                            partition_grad=True,
                                            initial_scale=128)
    # create data
    seed_all(2001 + local_rank)
    input_data = torch.randn(32, 128).cuda()

    zero1_output = zero1_model(input_data)
    zero2_output = zero2_model(input_data)
    assert torch.equal(zero1_output, zero2_output)

    # zero-dp backward
    zero1_optimizer.backward(zero1_output.mean().float(), sync_grad=False)
    zero2_optimizer.backward(zero2_output.mean().float(), sync_grad=False)

    for (n, z1p), z2p in zip(zero1_model.named_parameters(), zero2_model.parameters()):
        if z2p.grad is not None:
            # print(local_rank, n, z1p.shape, torch.max(z2p.grad), torch.max(torch.abs(z1p.grad - z2p.grad)))
            assert torch.equal(z1p.grad, z2p.grad)

    zero1_optimizer._sync_grad()
    zero2_optimizer._sync_grad()

    # step
    zero1_optimizer.step()
    zero2_optimizer.step()

    # check updated param
    for z1p, z2p in zip(zero1_model.parameters(), zero2_model.parameters()):
        assert torch.equal(z1p.data, z2p.data)


def exam_zero_1_torch_ddp():
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
    zero_model = MlpModel()
    torch_model = copy.deepcopy(zero_model)

    zero_model = zero_model.cuda().half()
    torch_model = DDP(torch_model.cuda(), bucket_cap_mb=0)
    torch_model = torch_model.cuda()

    # for (n, p), z1p in zip(torch_model.named_parameters(), zero_model.parameters()):
    #    half_close(p.data, z1p.data)

    # create optimizer
    zero_optimizer = torch.optim.SGD(zero_model.parameters(), lr=1)

    # we only test stage 1 here
    # in `check_sharded_param_consistency.py`, we will test whether
    # level 1 and 2 will produce exactly the same results
    zero_optimizer = LowLevelZeroOptimizer(zero_optimizer,
                                           overlap_communication=True,
                                           initial_scale=1,
                                           reduce_bucket_size=262144)

    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1)

    seed_all(1453 + local_rank)
    # create
    input_data = torch.rand(32, 128).cuda()

    # zero-dp forward
    zero_output = zero_model(input_data.half())

    # torch-ddp forward
    torch_output = torch_model(input_data)
    half_close(zero_output, torch_output, loose=True)

    # zero-dp backward
    zero_optimizer.backward(zero_output.mean().float(), sync_grad=False)

    # torch-ddp backward
    torch_output.mean().backward()

    # check grad
    for (n, p), z1p in zip(torch_model.named_parameters(), zero_model.parameters()):
        half_close(p.grad, z1p.grad, loose=True)

    # zero-dp step
    zero_optimizer._sync_grad()
    zero_optimizer.step()

    # torch ddp step
    torch_optimizer.step()

    # check updated param
    for (n, p), z1p in zip(torch_model.named_parameters(), zero_model.parameters()):
        # print(n, torch.max(torch.abs(p.data - z1p.data)))
        half_close(p.data, z1p.data, loose=True)


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')

    exam_zero_1_torch_ddp()
    exam_zero_1_2()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_zero_1_2():
    spawn(run_dist, 2)


if __name__ == '__main__':
    test_zero_1_2()
