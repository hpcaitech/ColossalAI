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
from _utils import tensor_equal, tensor_shard_equal


def init_1d_row(weight, bias):
    spec = TensorSpec(
        distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [-1], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        ParallelAction(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        weight.set_spec(spec)


def init_1d_col(weight, bias):
    spec = TensorSpec(
        distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [0], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        ParallelAction(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        weight.set_spec(spec)
        bias.set_spec(spec)


def run_with_spec(spec_init_func):
    with ColoInitContext(device=get_current_device()):
        model = torch.nn.Linear(4, 8)

    model_handy = copy(model)
    
    register_colo_module(torch.nn.Linear, ColoLinear())
    parallel_action = ParallelAction(ComputePattern.TP1D)
    if spec_init_func == init_1d_row:
        label = 'row'
    else:
        label = 'col'
    init_colo_module(model, parallel_action, recursive=True, label=label)
    
    spec_init_func(model_handy.weight, model_handy.bias)
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
    run_with_spec(init_1d_row)
    run_with_spec(init_1d_col)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_module_linear_1d(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_module_linear_1d(4)