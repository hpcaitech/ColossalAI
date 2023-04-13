import pytest
import torch
import torch.nn.functional as F
from torch.nn import Parameter

import colossalai
from colossalai.tensor import ColoTensor, ColoTensorSpec, ProcessGroup, ShardSpec
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device


def _run_layer_norm():
    ln_op = torch.nn.LayerNorm(2, 3, device=get_current_device())

    input_t = torch.randn(3, 2, device=get_current_device())

    pg = ProcessGroup(tp_degree=torch.distributed.get_world_size())
    input_t_colo = ColoTensor.from_torch_tensor(input_t.clone().detach(), ColoTensorSpec(pg))

    # prepare colossalai LN
    weight = ColoTensor(Parameter(ln_op.weight.detach()), ColoTensorSpec(pg))
    bias = ColoTensor(Parameter(ln_op.bias.detach()), ColoTensorSpec(pg))

    output = ln_op(input_t)
    output_colo = F.layer_norm(input_t_colo, ln_op.normalized_shape, weight, bias, ln_op.eps)

    assert torch.allclose(output_colo, output)

    torch.mean(output).backward()
    torch.mean(output_colo).backward()

    assert torch.allclose(ln_op.weight.grad, weight.grad)


def check_spec_eq(tensor, other):
    assert isinstance(tensor, ColoTensor) and isinstance(other, ColoTensor)
    for k in dir(tensor.dist_spec):
        if not k.startswith('__'):
            assert hasattr(other.dist_spec, k), f"{k}"
            assert getattr(tensor.dist_spec, k) == getattr(other.dist_spec, k)


def check_element_wise_ops():
    world_size = torch.distributed.get_world_size()
    pg = ProcessGroup(tp_degree=world_size)
    t = torch.rand(2, 2)
    x = ColoTensor(t, spec=ColoTensorSpec(pg, ShardSpec([0], [pg.tp_world_size()])))

    check_spec_eq(x, x.cuda())
    assert torch.equal(x.cuda(), t.cuda())
    check_spec_eq(x, torch.abs(x))
    assert torch.equal(torch.abs(x), torch.abs(t))
    check_spec_eq(x, F.sigmoid(x))
    assert torch.equal(F.sigmoid(x), F.sigmoid(t))


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    check_element_wise_ops()
    _run_layer_norm()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [2])
@rerun_if_address_is_in_use()
def test_element_wise_ops(world_size):
    spawn(run_dist, world_size)


def run_dist2(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    _run_layer_norm()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1])
@rerun_if_address_is_in_use()
def test_ln(world_size):
    spawn(run_dist2, world_size)


def check_all():
    test_element_wise_ops(2)


if __name__ == '__main__':
    check_all()
