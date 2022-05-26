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
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction, DistSpecManager, register_colo_module, init_colo_module, ColoLinear
from _utils import tensor_equal, tensor_shard_equal, set_seed
from tests.components_to_test.registry import non_distributed_component_funcs

def run_simplenet_with_spec(label):
    get_components_func = non_distributed_component_funcs.get_callable('simple_net')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    set_seed(1)
    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=True)
    
    if rank == 0:
        model_seq = model_builder(checkpoint=True)
        model_seq = model_seq.cuda()

        # Make two models have the same init params
        for p1, p2 in zip(model.parameters(), model_seq.parameters()):
            p2.data.copy_(p1.data)

    parallel_action = ParallelAction(ComputePattern.TP1D)
    init_colo_module(model, parallel_action, recursive=True, label=label)

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

def run_linear_with_spec(label):
    with ColoInitContext(device=get_current_device()):
        model = torch.nn.Linear(4, 8)

    model_handy = copy(model)
    
    parallel_action = ParallelAction(ComputePattern.TP1D)
    init_colo_module(model, parallel_action, recursive=True, label=label)
    
    x = torch.rand(2, 4).cuda()
    out = model(x)
    colo_out = model_handy(x)
    assert tensor_equal(out, colo_out)
    grad = torch.rand_like(out)
    out.backward(grad)
    colo_out.backward(grad)
    assert tensor_shard_equal(model.weight.grad, model_handy.weight.grad)
    assert tensor_shard_equal(model.bias.grad, model_handy.bias.grad)


def run_dist(rank, world_size, port, func):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    func('col')
    func('row')
    func('default')


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_module_linear_1d(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port(), func=run_linear_with_spec)
    mp.spawn(run_func, nprocs=world_size)

@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_module_simplenet(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port(), func=run_simplenet_with_spec)
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_module_simplenet(4)