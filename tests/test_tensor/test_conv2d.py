import torch
from colossalai.tensor import ColoTensor, distspec, ProcessGroup

from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.tensor import TensorSpec, ComputePattern, ComputeSpec, DistSpecManager
from _utils import tensor_equal, tensor_shard_equal


def init_1d_row(weight, bias, pg: ProcessGroup):
    # split on the last dim.
    spec = TensorSpec(distspec.shard(pg, [-1], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        weight.set_tensor_spec(spec)
        bias.set_tensor_spec(spec)


def run_with_spec(spec_init_func):
    # initialize a ProcessGroup instance
    pg = ProcessGroup(tp_degree=torch.distributed.get_world_size())

    # initialize a model with only a conv 2d layer
    model = torch.nn.Conv2d(4, 4, 3, stride=2).cuda()
    weight = ColoTensor(torch.nn.Parameter(model.weight.detach()))
    bias = ColoTensor(torch.nn.Parameter(model.bias.detach()))

    # set tensor parallel spec. for the model
    spec_init_func(weight, bias)

    # fwd the torch model
    x = torch.rand(2, 4).cuda()
    out = model(x)

    # compute the colossalai function
    colo_out = F.conv2d(x, weight, bias)
    colo_out = colo_out.to_replicate()

    # check fwd results
    assert tensor_equal(out, colo_out)

    # check bwd results
    grad = torch.rand_like(out)
    out.backward(grad)
    colo_out.backward(grad)
    assert tensor_shard_equal(model.weight.grad, weight.grad)
    assert tensor_shard_equal(model.bias.grad, bias.grad)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_with_spec(init_1d_row)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_conv2d(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_conv2d(4)
