import copy
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.utils import free_port
from colossalai.zero import LowLevelZeroOptimizer


def check_equal(a, b, rtol=1e-4, atol=1e-3):
    """
    This function checks if two tensors are equal within tolerance
    """
    assert torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol), f'a = {a}, b = {b}'


def check_completely_equal(a, b):
    """
    This function checks if two tensors are completely equal
    """
    assert torch.all(a == b), f'a = {a}, b = {b}'


class TestModel(nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 512)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def exam_zero_1_2_grad_clip():
    # create model
    zero1_model = TestModel().cuda().half()
    zero2_model = copy.deepcopy(zero1_model)

    # create optimizer
    zero1_optimizer = torch.optim.Adam(zero1_model.parameters(), lr=0.001)
    zero2_optimizer = torch.optim.Adam(zero2_model.parameters(), lr=0.001)
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

    # create
    input_data = torch.rand(32, 128).cuda().half()

    # forward
    zero1_output = zero1_model(input_data)
    zero2_output = zero2_model(input_data)
    check_completely_equal(zero1_output, zero2_output)

    # backward
    zero1_optimizer.backward(zero1_output.mean().float())
    zero2_optimizer.backward(zero2_output.mean().float())

    # check grad
    # as this param is small, the backward reduction
    # will not be fired
    for z1p, z2p in zip(zero1_model.parameters(), zero2_model.parameters()):
        check_completely_equal(z1p.grad, z2p.grad)

    # step
    zero1_optimizer.sync_grad()
    zero2_optimizer.sync_grad()

    # step
    zero1_optimizer.step()
    zero2_optimizer.step()

    # check updated param
    for z1p, z2p in zip(zero1_model.parameters(), zero2_model.parameters()):
        check_completely_equal(z1p.data, z2p.data)


def exam_zero_1_grad_clip():
    # create models
    zero_model = TestModel()
    torch_model = copy.deepcopy(zero_model)

    zero_model = zero_model.cuda().half()
    torch_model = DDP(torch_model.cuda())

    # create optimizer
    zero_optimizer = torch.optim.Adam(zero_model.parameters(), lr=0.001)

    # we only test stage 1 here
    # in `check_sharded_param_consistency.py`, we will test whether
    # level 1 and 2 will produce exactly the same results
    zero_optimizer = LowLevelZeroOptimizer(zero_optimizer,
                                           overlap_communication=True,
                                           initial_scale=1,
                                           clip_grad_norm=1.0)

    torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001)

    # create
    input_data = torch.rand(32, 128).cuda()

    # zero-dp forward
    zero_output = zero_model(input_data.half())

    # torch-ddp forward
    torch_output = torch_model(input_data)
    check_equal(zero_output, torch_output)

    # zero-dp backward
    zero_optimizer.backward(zero_output.mean().float())

    # torch-ddp backward
    torch_output.mean().backward()

    # check grad
    for p, z1p in zip(torch_model.parameters(), zero_model.parameters()):
        check_equal(p.grad, z1p.grad)

    # zero-dp step
    zero_optimizer.sync_grad()
    zero_optimizer.step()

    # torch ddp step
    torch.nn.utils.clip_grad_norm_(torch_model.parameters(), 1.0)
    torch_optimizer.step()

    # check updated param
    for p, z1p in zip(torch_model.parameters(), zero_model.parameters()):
        check_equal(p.data, z1p.data, atol=5e-4)


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')

    exam_zero_1_2_grad_clip()
    exam_zero_1_grad_clip()


@pytest.mark.dist
def test_grad_clip():
    world_size = 2
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_grad_clip()
