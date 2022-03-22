import torch
import colossalai
import copy
import pytest
import torch.multiprocessing as mp
from colossalai.zero import ShardedOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from colossalai.utils import free_port
from functools import partial
from common import allclose
from tests.components_to_test.registry import non_distributed_component_funcs


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
    test_models = ['repeated_computed_layers', 'resnet18', 'nested_model']

    for name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(name)
        model_builder, train_dataloader, *_ = get_components_func()

        # create model
        oss_model = model_builder(checkpoint=True).cuda().half()
        pg_model = copy.deepcopy(oss_model)

        # create optimizer
        oss_optimizer = torch.optim.Adam(oss_model.parameters(), lr=0.001)
        pg_optimizer = torch.optim.Adam(pg_model.parameters(), lr=0.001)
        oss_optimizer = ShardedOptimizer(oss_optimizer, overlap_communication=True, initial_scale=1, clip_grad_norm=0.0)
        pg_optimizer = ShardedOptimizer(pg_optimizer,
                                        overlap_communication=True,
                                        partition_grad=True,
                                        initial_scale=1,
                                        clip_grad_norm=0.0)

        # create
        data, label = next(iter(train_dataloader))
        input_data = data.cuda().half()

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
        for oss_param, pg_param in zip(oss_model.parameters(), pg_model.parameters()):
            check_completely_equal(oss_param.grad, pg_param.grad)

        # step
        oss_optimizer.sync_grad()
        pg_optimizer.sync_grad()

        # step
        oss_optimizer.step()
        pg_optimizer.step()

        # check updated param
        for oss_param, pg_param in zip(oss_model.parameters(), pg_model.parameters()):
            check_completely_equal(oss_param, pg_param)


def check_sharded_optim_against_torch_ddp():
    """
    In this test, two pairs of model and optimizers are created.
    1. zero: use sharded optimizer and fp16 parameters
    2. torch: use torch DDP and fp32 parameters

    We feed these two sets of models with the same input and check if the
    differences in model output and updated parameters are within tolerance.
    """

    test_models = ['repeated_computed_layers', 'resnet18', 'nested_model']

    for name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(name)
        model_builder, train_dataloader, *_ = get_components_func()

        # create model
        zero_model = model_builder(checkpoint=True).cuda()
        torch_model = copy.deepcopy(zero_model)

        zero_model = zero_model.half()
        torch_model = DDP(torch_model.cuda())

        # create optimizer
        zero_optimizer = torch.optim.Adam(zero_model.parameters(), lr=0.001)

        # we only test stage 1 here
        # in `check_sharded_param_consistency.py`, we will test whether
        # level 1 and 2 will produce exactly the same results
        zero_optimizer = ShardedOptimizer(zero_optimizer,
                                          overlap_communication=True,
                                          initial_scale=1,
                                          clip_grad_norm=0.0)
        torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001)

        # create
        input_data, _ = next(iter(train_dataloader))
        input_data = input_data.cuda()

        # zero-dp forward
        zero_output = zero_model(input_data.half())

        # torch-ddp forward
        torch_output = torch_model(input_data)
        allclose(zero_output, torch_output.half())

        # zero-dp backward
        zero_optimizer.backward(zero_output.mean().float())

        # torch-ddp backward
        torch_output.mean().backward()

        # check grad
        for oss_param, torch_param in zip(zero_model.parameters(), torch_model.parameters()):
            allclose(oss_param.grad, torch_param.grad.half())

        # zero-dp step
        zero_optimizer.sync_grad()
        zero_optimizer.step()

        # torch ddp step
        torch_optimizer.step()

        # check updated param
        for oss_param, torch_param in zip(zero_model.parameters(), torch_model.parameters()):
            allclose(oss_param, torch_param.half())


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
