import torch
import pytest
import colossalai
import torch.nn.functional as F
import torch.multiprocessing as mp
from functools import partial
from colossalai.tensor import ColoTensor, ColoParameter
from colossalai.utils import get_current_device
from torch.nn import Parameter
from torch.distributed.distributed_c10d import _get_default_group
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.tensor import distspec, TensorSpec


def test_layernorm():
    ln_op = torch.nn.LayerNorm(2, 3, device=get_current_device())

    input_t = torch.randn(3, 2, device=get_current_device())
    input_t_colo = ColoTensor.from_torch_tensor(input_t.clone().detach())

    # prepare colossalai LN
    weight = ColoTensor(Parameter(ln_op.weight.detach()))
    bias = ColoTensor(Parameter(ln_op.bias.detach()))

    output = ln_op(input_t)
    output_colo = F.layer_norm(input_t_colo, ln_op.normalized_shape, weight, bias, ln_op.eps)

    assert torch.allclose(output_colo, output)

    torch.mean(output).backward()
    torch.mean(output_colo).backward()

    assert torch.allclose(ln_op.weight.grad, weight.grad)


def check_spec_eq(tensor, other):
    assert isinstance(tensor, ColoTensor) and isinstance(other, ColoTensor)
    for k in dir(tensor.tensor_spec.dist_spec):
        if not k.startswith('__'):
            assert hasattr(other.tensor_spec.dist_spec, k)
            assert getattr(tensor.tensor_spec.dist_spec, k) == getattr(other.tensor_spec.dist_spec, k)


def check_element_wise_ops():
    pg = _get_default_group()
    t = torch.rand(2, 2)
    x = ColoTensor(t, spec=TensorSpec(distspec.shard(pg, [0], [pg.size()])))
    check_spec_eq(x, x.cuda())
    assert torch.equal(x.cuda(), t.cuda())
    check_spec_eq(x, torch.abs(x))
    assert torch.equal(torch.abs(x), torch.abs(t))
    check_spec_eq(x, F.sigmoid(x))
    assert torch.equal(F.sigmoid(x), F.sigmoid(t))


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    check_element_wise_ops()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [2])
@rerun_if_address_is_in_use()
def test_element_wise_ops(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


def check_all():
    test_layernorm()
    test_element_wise_ops(2)


if __name__ == '__main__':
    check_all()
