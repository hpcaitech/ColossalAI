import torch
from colossalai.tensor import ColoTensor, ColoParameter
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


def check_all():
    test_layernorm()


if __name__ == '__main__':
    check_all()
