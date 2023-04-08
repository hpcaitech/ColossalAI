import pytest
import torch
import torch.nn as nn

import colossalai
from colossalai.tensor import ColoTensor, ColoTensorSpec, ProcessGroup
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.test_tensor.common_utils import split_param_col_tp1d, split_param_row_tp1d, tensor_equal, tensor_shard_equal


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


def run_with_spec(spec_init_func, split_bias):
    model = Conv1D(4, 16).cuda()
    world_size = torch.distributed.get_world_size()
    pg = ProcessGroup(tp_degree=world_size)

    weight = ColoTensor(torch.nn.Parameter(model.weight.detach()), ColoTensorSpec(pg))
    bias = ColoTensor(torch.nn.Parameter(model.bias.detach()), ColoTensorSpec(pg))

    spec_init_func(weight, pg)
    if split_bias:
        spec_init_func(bias, pg)

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
    run_with_spec(spec_init_func=split_param_row_tp1d, split_bias=False)
    run_with_spec(spec_init_func=split_param_col_tp1d, split_bias=True)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_addmm_1d(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_addmm_1d(4)
