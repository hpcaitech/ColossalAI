from tests.components_to_test.registry import non_distributed_component_funcs

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils import ColoInitContext
from colossalai.tensor import named_params_with_colotensor, TensorSpec, ComputePattern, ParallelAction, ColoTensor, ColoOptimizer
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc

from functools import partial
import random
import os
import numpy as np


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_1d_col_tp():
    # A simple net with two stacked nn.Linear
    get_components_func = non_distributed_component_funcs.get_callable('simple_net')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    set_seed(1)
    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=True)

    parallel_action_list_row = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DRow_Linear, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec_row = TensorSpec(parallel_action_list_row)

    parallel_action_list_col = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DCol_Linear, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec_col = TensorSpec(parallel_action_list_col)

    parallel_action_list_embedding_col = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DCol_Embedding, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec_embedding_col = TensorSpec(parallel_action_list_embedding_col)

    set_seed(1)
    if rank == 0:
        model_torch = model_builder(checkpoint=True)
        model_torch = model_torch.cuda()

    # A naive way to set spec for all weights in Linear
    for name, p in named_params_with_colotensor(model):
        if not isinstance(p, ColoTensor):
            continue
        if 'proj1' in name and ('weight' in name or 'bias' in name):
            p.set_spec(spec_col)
        if 'proj2' in name and 'weight' in name:
            p.set_spec(spec_row)
        if 'embed' in name and 'weight' in name:
            p.set_spec(spec_embedding_col)

    model = model.cuda()

    for i, (data, label) in enumerate(train_dataloader):
        data = data.to(get_current_device())
        label = label.to(get_current_device())

        torch.distributed.broadcast(data, 0, group=gpc.get_group(ParallelMode.PARALLEL_1D))
        torch.distributed.broadcast(label, 0, group=gpc.get_group(ParallelMode.PARALLEL_1D))

        # Bcast rank0 data to all processes
        if criterion:
            output = model(data)
            loss = criterion(output, label)
        else:
            output = model(data, label)
            loss = output

        # For reference
        if rank == 0:
            if criterion:
                output_torch = model_torch(data)
                loss_torch = criterion(output_torch, label)
            else:
                output_torch = model_torch(data, label)
                loss_torch = output_torch

        if rank == 0:
            # print(loss.torch_tensor().item())
            # print('loss torch', loss_torch.item())
            assert torch.allclose(loss.torch_tensor(), loss_torch, rtol=1e-2)

        loss.backward()

        if rank == 0:
            loss_torch.backward()
        if i > 5:
            break


# Test the overrided parameters() and named_parameters() member functions
def test_model_parameters():
    # build a module with 2 Linear, 4 parameters in total.
    class Net(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.fcs = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 2))
            self.extra_param = torch.nn.Parameter(torch.randn(2))

    with ColoInitContext(device=get_current_device()):
        model = Net()

    param_cnt = 0
    for name, p in model.named_parameters():
        param_cnt += 1
    assert param_cnt == 5

    param_cnt = 0
    for name, p in model.named_parameters(recurse=False):
        param_cnt += 1
    assert param_cnt == 1

    param_cnt = 0
    for p in model.fcs[0].parameters(recurse=False):
        param_cnt += 1
    assert param_cnt == 2


def test_colo_optimizer():
    get_components_func = non_distributed_component_funcs.get_callable('simple_net')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    set_seed(1)
    with ColoInitContext(lazy_memory_allocate=False, device=get_current_device()):
        model = model_builder(checkpoint=True)

    colo_optimizer = ColoOptimizer(dict(model.named_parameters()), torch.optim.SGD, lr=0.1)
    for i, (data, label) in enumerate(train_dataloader):
        colo_optimizer.zero_grad()
        data = data.to(get_current_device())
        label = label.to(get_current_device())

        # Bcast rank0 data to all processes
        if criterion:
            output = model(data)
            loss = criterion(output, label)
        else:
            output = model(data, label)
            loss = output

        loss.backward()
        colo_optimizer.step()

        if i > 5:
            break


def run_1d_row_tp():
    # A simple net with two stacked nn.Linear
    get_components_func = non_distributed_component_funcs.get_callable('simple_net')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    set_seed(1)
    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=True)

    parallel_action_list = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DRow_Linear, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec = TensorSpec(parallel_action_list)

    parallel_action_list_embedding_row = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DRow_Embedding, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec_embedding_row = TensorSpec(parallel_action_list_embedding_row)

    set_seed(1)
    if rank == 0:
        model_torch = model_builder(checkpoint=True)
        model_torch = model_torch.cuda()

    # A naive way to set spec for all weights in Linear
    for name, p in model.colo_named_parameters():
        if not isinstance(p, ColoTensor):
            continue
        if 'weight' in name and 'LayerNorm' not in name and 'ln' not in name and 'embed' not in name:
            p.set_spec(spec)
        if 'embed' in name and 'weight' in name:
            p.set_spec(spec_embedding_row)

    model = model.cuda()

    for i, (data, label) in enumerate(train_dataloader):
        data = data.to(get_current_device())
        label = label.to(get_current_device())

        torch.distributed.broadcast(data, 0, group=gpc.get_group(ParallelMode.PARALLEL_1D))
        torch.distributed.broadcast(label, 0, group=gpc.get_group(ParallelMode.PARALLEL_1D))

        # Bcast rank0 data to all processes
        if criterion:
            output = model(data)
            loss = criterion(output, label)
        else:
            output = model(data, label)
            loss = output

        # For reference
        if rank == 0:
            if criterion:
                output_torch = model_torch(data)
                loss_torch = criterion(output_torch, label)
            else:
                output_torch = model_torch(data, label)
                loss_torch = output_torch

        if rank == 0:
            # print(loss.torch_tensor().item())
            # print('loss torch', loss_torch.item())
            assert torch.allclose(loss.torch_tensor(), loss_torch, rtol=1e-2)

        loss.backward()

        if rank == 0:
            loss_torch.backward()
        if i > 5:
            break


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_1d_row_tp()
    run_1d_col_tp()

@pytest.mark.dist
@parameterize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_simple_net(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_simple_net()
    # test_model_parameters()
    # test_colo_optimizer()
