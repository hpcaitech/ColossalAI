from copy import copy
from colossalai.utils.cuda import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
import torch
from colossalai.context.parallel_mode import ParallelMode
from colossalai.tensor import ColoTensor, distspec

from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction, DistSpecManager, register_colo_module, init_colo_module
from _utils import tensor_equal, tensor_shard_equal, set_seed
from tests.components_to_test.registry import non_distributed_component_funcs

def run_model_with_spec(mode, model_name):
    print(f'model_name: {model_name} | mode: {mode}')
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    set_seed(1)
    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=False)
    
    if rank == 0:
        model_seq = model_builder(checkpoint=False)
        model_seq = model_seq.cuda()

        # Make two models have the same init params
        for p1, p2 in zip(model.parameters(), model_seq.parameters()):
            p2.data.copy_(p1.data)

    parallel_action = ParallelAction(ComputePattern.TP1D)
    init_colo_module(model, parallel_action, recursive=True, mode=mode)

    model = model.cuda()
    for i, (data, label) in enumerate(train_dataloader):
        data = data.to(get_current_device())
        label = label.to(get_current_device())

        torch.distributed.broadcast(data, 0, group=gpc.get_group(ParallelMode.PARALLEL_1D))
        torch.distributed.broadcast(label, 0, group=gpc.get_group(ParallelMode.PARALLEL_1D))

        if criterion:
            output = model(data)
            loss = criterion(output, label)
        else:
            output = model(data, label)
            loss = output

        # For reference
        if rank == 0:
            if criterion:
                output_seq = model_seq(data)
                loss_seq = criterion(output_seq, label)
            else:
                output_seq = model_seq(data, label)
                loss_seq = output_seq

        if rank == 0:
            with torch.no_grad():
                assert torch.allclose(loss, loss_seq, rtol=1e-2)

        loss.backward()

        if rank == 0:
            loss_seq.backward()

            with torch.no_grad():
                # check param
                for p1, p2 in zip(model.parameters(), model_seq.parameters()):
                    if p1.size() == p2.size():
                        assert torch.allclose(p1, p2)
                    else:
                        if p1.size(-1) < p2.size(-1):    # col
                            world_size = p2.size(-1) // p1.size(-1)
                            split_p2 = torch.chunk(p2, world_size, dim=-1)[0]

                        elif p1.size(0) < p2.size(0):    # row
                            world_size = p2.size(0) // p1.size(0)
                            split_p2 = torch.chunk(p2, world_size, dim=0)[0]

                        assert torch.allclose(p1, split_p2)

        if i > 3:
            break

def run_linear_with_spec(mode):
    with ColoInitContext(device=get_current_device()):
        model = torch.nn.Linear(4, 8)

    model_handy = copy(model)
    
    parallel_action = ParallelAction(ComputePattern.TP1D)
    init_colo_module(model, parallel_action, recursive=True, mode=mode)
    
    x = torch.rand(2, 4).cuda()
    out = model(x)
    colo_out = model_handy(x)
    assert tensor_equal(out, colo_out)
    grad = torch.rand_like(out)
    out.backward(grad)
    colo_out.backward(grad)
    assert tensor_shard_equal(model.weight.grad, model_handy.weight.grad)
    assert tensor_shard_equal(model.bias.grad, model_handy.bias.grad)


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_linear_with_spec('col')
    run_linear_with_spec('row')

def run_dist_model(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    for model_name in ['simple_net', 'bert']:
        run_model_with_spec('col', model_name)
        if 'simple_net' == model_name:
            # Bert in our testcase is a classification model returning 0 or 1.
            # row shard for all layers is invalid because the first dim of some layer is the classification type size 2.
            run_model_with_spec('row', model_name)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_module_linear_1d(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)

@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_module_model(world_size):
    run_func = partial(run_dist_model, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_module_model(4)