import colossalai
import torch
import pytest
import torch.nn as nn
import torch.multiprocessing as mp
from colossalai.tensor import dist_spec
from colossalai.utils import ColoInitContext
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction
from colossalai.context import ParallelMode
from colossalai.utils.cuda import get_current_device
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from functools import partial
from colossalai.core import global_context as gpc


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


def init_1d_row(model):
    spec = TensorSpec(
        dist_spec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [0], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        [ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DRow, parallel_mode=ParallelMode.PARALLEL_1D)])
    with dist_spec.DistSpecManager.no_grad():
        for n, p in model.colo_named_parameters():
            if 'weight' in n:
                p.set_spec(spec, init_pg=True)


def init_1d_col(model):
    spec = TensorSpec(
        dist_spec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [-1], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        [ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DCol, parallel_mode=ParallelMode.PARALLEL_1D)])
    with dist_spec.DistSpecManager.no_grad():
        for n, p in model.colo_named_parameters():
            p.set_spec(spec, init_pg=True)


def run_with_spec(spec_init_func):
    with ColoInitContext(device=get_current_device()):
        model = Conv1D(4, 16)
    weight = model.weight.torch_tensor().clone()
    bias = model.bias.torch_tensor().clone()
    spec_init_func(model)
    x = torch.rand(2, 16).cuda()
    out = model(x)
    assert torch.allclose(out.torch_tensor(), torch.addmm(bias, x, weight))


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_with_spec(init_1d_row)
    run_with_spec(init_1d_col)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2, 4])
@rerun_if_address_is_in_use()
def test_addmm_1d(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_addmm_1d(2)
