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


def check_equal(a, b):
    """
    This function checks if two tensors are equal within tolerance
    """
    assert torch.allclose(a.float(), b.float(), rtol=1e-4, atol=1e-3), f'a = {a}, b = {b}'


def check_completely_equal(a, b):
    """
    This function checks if two tensors are completely equal
    """
    assert torch.all(a == b), f'a = {a}, b = {b}'


def check_sharded_param_consistency():
    """
    In this test, we want to test whether zero stage 1 and 2
    deliver the same numerical results despite different communication
    pattern

    we use these prefixes to differentiate the zero stage
    oss: partition optimizer states
    pg: partition gradients and optimizer states

    """

    # create layers
    oss_linear1 = nn.Linear(128, 256)
    oss_linear2 = nn.Linear(256, 512)

    # create model
    oss_model = nn.Sequential(oss_linear1, oss_linear2)
    pg_model = copy.deepcopy(oss_model)

    oss_model = oss_model.cuda().half()
    pg_model = pg_model.cuda().half()

    # create optimizer
    oss_optimizer = torch.optim.Adam(oss_model.parameters(), lr=0.001)
    pg_optimizer = torch.optim.Adam(pg_model.parameters(), lr=0.001)
    oss_optimizer = LowLevelZeroOptimizer(oss_optimizer,
                                          overlap_communication=True,
                                          initial_scale=1,
                                          clip_grad_norm=0.0)
    pg_optimizer = LowLevelZeroOptimizer(pg_optimizer,
                                         overlap_communication=True,
                                         partition_grad=True,
                                         initial_scale=1,
                                         clip_grad_norm=0.0)

    # create
    input_data = torch.rand(32, 128).cuda().half()

    # forward
    oss_output = oss_model(input_data)
    pg_output = pg_model(input_data)
    check_completely_equal(oss_output, pg_output)

    # backward
    oss_optimizer.backward(oss_output.mean().float())
    pg_optimizer.backward(pg_output.mean().float())

    # check grad
    # as this param is small, the backward reduction
    # will not be fired
    oss_linear1_grad = oss_model[0].weight.grad
    oss_linear2_grad = oss_model[1].weight.grad
    pg_linear1_grad = pg_model[0].weight.grad
    pg_linear2_grad = pg_model[1].weight.grad
    check_completely_equal(oss_linear1_grad, pg_linear1_grad)
    check_completely_equal(oss_linear2_grad, pg_linear2_grad)

    # step
    oss_optimizer.sync_grad()
    pg_optimizer.sync_grad()

    # step
    oss_optimizer.step()
    pg_optimizer.step()

    # check updated param
    check_completely_equal(oss_model[0].weight, pg_model[0].weight)
    check_completely_equal(oss_model[1].weight, pg_model[1].weight)


def check_sharded_optim_against_torch_ddp():
    """
    In this test, two pairs of model and optimizers are created.
    1. zero: use sharded optimizer and fp16 parameters
    2. torch: use torch DDP and fp32 parameters

    We feed these two sets of models with the same input and check if the
    differences in model output and updated parameters are within tolerance.
    """

    # create layer
    zero_linear1 = nn.Linear(128, 256)
    zero_linear2 = nn.Linear(256, 512)

    # create model
    zero_model = nn.Sequential(zero_linear1, zero_linear2)
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
                                           clip_grad_norm=0.0)

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
    zero_linear1_grad = zero_model[0].weight.grad
    zero_linear2_grad = zero_model[1].weight.grad
    torch_linear1_grad = torch_model.module[0].weight.grad
    torch_linear2_grad = torch_model.module[1].weight.grad
    check_equal(zero_linear1_grad, torch_linear1_grad)
    check_equal(zero_linear2_grad, torch_linear2_grad)

    # zero-dp step
    zero_optimizer.sync_grad()
    zero_optimizer.step()

    # torch ddp step
    torch_optimizer.step()

    # check updated param
    check_equal(zero_model[0].weight, torch_model.module[0].weight)
    check_equal(zero_model[1].weight, torch_model.module[1].weight)


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')

    check_sharded_optim_against_torch_ddp()
    check_sharded_param_consistency()


@pytest.mark.dist
def test_sharded_optim():
    world_size = 2
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_sharded_optim()
