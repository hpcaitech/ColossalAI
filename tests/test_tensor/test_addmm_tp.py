import colossalai
import torch
import pytest
import torch.nn as nn
import torch.multiprocessing as mp
from colossalai.tensor import ColoTensor, ProcessGroup
from colossalai.tensor import distspec
from colossalai.tensor import TensorSpec, ComputePattern, ComputeSpec, DistSpecManager
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from functools import partial
from _utils import tensor_shard_equal, tensor_equal


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.ones(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


def init_1d_row(weight, bias, pg: ProcessGroup):
    spec = TensorSpec(distspec.shard(pg, [0], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        weight.set_tensor_spec(spec)


def init_1d_col(weight, bias, pg: ProcessGroup):
    spec = TensorSpec(distspec.shard(pg, [-1], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        weight.set_tensor_spec(spec)
        bias.set_tensor_spec(spec)


def run_with_spec(spec_init_func):
    model = Conv1D(4, 16).cuda()
    weight = ColoTensor(torch.nn.Parameter(model.weight.detach()))
    bias = ColoTensor(torch.nn.Parameter(model.bias.detach()))
    world_size = torch.distributed.get_world_size()
    pg = ProcessGroup(tp_degree=world_size)
    spec_init_func(weight, bias, pg)
    x = torch.rand(2, 16).cuda()
    out = model(x)
    colo_out = torch.addmm(bias, x, weight)
    colo_out = colo_out.to_replicate()
    assert tensor_equal(out, colo_out)
    grad = torch.rand_like(out)
    out.backward(grad)
    colo_out.backward(grad)
    tensor_shard_equal(model.weight.grad, weight.grad, pg.tp_local_rank(), pg.tp_world_size())
    tensor_shard_equal(model.bias.grad, bias.grad, pg.tp_local_rank(), pg.tp_world_size())


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_with_spec(init_1d_row)
    run_with_spec(init_1d_col)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_addmm_1d(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_addmm_1d(4)
