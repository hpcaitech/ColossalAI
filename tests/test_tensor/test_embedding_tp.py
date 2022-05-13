import torch
from colossalai.context.parallel_mode import ParallelMode
from colossalai.tensor import ColoTensor
from torch.nn import functional as F
from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction, dist_spec, DistSpecManager


def init_1d_row(weight):
    spec = TensorSpec(
        dist_spec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [0], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        [ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DRow, parallel_mode=ParallelMode.PARALLEL_1D)])
    with DistSpecManager.no_grad():
        weight.set_spec(spec)


def check_grad_1d_row(model: torch.nn.Module, weight):
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
    size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    assert torch.allclose(model.weight.grad.chunk(size, 0)[rank], weight.grad)


def init_1d_col(weight):
    spec = TensorSpec(
        dist_spec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [-1], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        [ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DCol, parallel_mode=ParallelMode.PARALLEL_1D)])
    with DistSpecManager.no_grad():
        weight.set_spec(spec)


def check_grad_1d_col(model: torch.nn.Module, weight):
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
    size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    assert torch.allclose(model.weight.grad.chunk(size, -1)[rank], weight.grad)


def run_with_spec(spec_init_func, check_grad_func):
    model = torch.nn.Embedding(12, 32).cuda()
    weight = ColoTensor.init_from_torch_tensor(torch.nn.Parameter(model.weight.detach()))
    spec_init_func(weight)
    x = torch.tensor((0, 3, 6, 9)).cuda()
    out = model(x)
    colo_out = F.embedding(x, weight)
    assert torch.allclose(out, colo_out)
    grad = torch.rand_like(out)
    out.backward(grad)
    colo_out.backward(grad)
    check_grad_func(model, weight)


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_with_spec(init_1d_row, check_grad_1d_row)
    run_with_spec(init_1d_col, check_grad_1d_col)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_embedding_1d(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_embedding_1d(4)
