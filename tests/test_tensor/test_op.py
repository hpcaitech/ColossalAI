import torch
import pytest
from colossalai.tensor import ColoTensor, ColoParameter
from copy import deepcopy
from colossalai.utils import get_current_device
from torch.nn import Parameter
import torch.nn.functional as F


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


# The test case failed
# def test_uniform():
#     t = ColoTensor(torch.zeros(3, 5))
#     torch.nn.init.uniform_(t)
#     print(t)


@pytest.mark.skip
def test_element_wise():
    t_ref = torch.randn(3, 5)
    t = ColoTensor.init_from_torch_tensor(t_ref.clone())
    assert torch.mean(t) == torch.mean(t_ref)
    assert allclose(torch.nn.functional.gelu(t).torch_tensor(), torch.nn.functional.gelu(t_ref))
    assert allclose(torch.nn.functional.relu(t).torch_tensor(), torch.nn.functional.relu(t_ref))


# Test a function not wrapped by
@pytest.mark.skip
def test_no_wrap_op():
    t_ref = torch.randn(3, 5)
    t = ColoTensor.init_from_torch_tensor(t_ref.clone())
    assert torch.sum(t) == torch.sum(t_ref)
    assert torch.sum(input=t) == torch.sum(input=t_ref)


def check_all():
    test_element_wise()
    test_no_wrap_op()


if __name__ == '__main__':
    check_all()
