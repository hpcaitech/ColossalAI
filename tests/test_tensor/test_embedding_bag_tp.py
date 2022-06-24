import torch
from colossalai.context.parallel_mode import ParallelMode
from colossalai.tensor import ColoTensor, distspec, ColoParameter
from torch.nn import functional as F
from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
from colossalai.tensor import TensorSpec, ComputePattern, ComputeSpec, DistSpecManager
from _utils import tensor_equal, tensor_shard_equal


def init_1d_col(weight):
    spec = TensorSpec(
        distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [-1], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        ComputeSpec(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        weight.set_tensor_spec(spec)


def run_with_spec(spec_init_func):
    model = torch.nn.EmbeddingBag(10, 4).cuda()
    weight = ColoParameter(model.weight.clone())
    spec_init_func(weight)
    inputs = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]).cuda()
    offsets = torch.tensor([0, 4]).cuda()
    out = model(inputs, offsets=offsets)
    colo_out = F.embedding_bag(inputs, weight, offsets=offsets)
    assert tensor_equal(out, colo_out)
    grad = torch.rand_like(out)
    out.backward(grad)
    colo_out.backward(grad)
    assert tensor_shard_equal(model.weight.grad, weight.grad)


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_with_spec(init_1d_col)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_embedding_bag_1d(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_embedding_bag_1d(4)
