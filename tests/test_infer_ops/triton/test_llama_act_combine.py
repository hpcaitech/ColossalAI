import pytest
import torch
from packaging import version
from torch import nn

from colossalai.kernel.triton.llama_act_combine_kernel import LlamaActCombine

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")
TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.4')

BATCH_SIZE = 4
SEQ_LEN = 16
HIDDEN_SIZE = 32


def SwiGLU(x):
    """Gated linear unit activation function.
    Args:
        x : input array
        axis: the axis along which the split should be computed (default: -1)
    """
    size = x.shape[-1]
    assert size % 2 == 0, "axis size must be divisible by 2"
    x1, x2 = torch.split(x, size // 2, -1)
    return x1 * (x2 * torch.sigmoid(x2.to(torch.float32)).to(x.dtype))


@pytest.mark.skipif(not (HAS_TRITON and TRITON_CUDA_SUPPORT), reason="requires triton")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_llama_act_combine(dtype: str):
    x_gate = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE * 2, dtype=dtype).cuda()
    x_gate_torch = nn.Parameter(x_gate.detach().clone())
    x_gate_kernel = nn.Parameter(x_gate.detach().clone())
    x_up = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=dtype).cuda()
    x_up_torch = nn.Parameter(x_up.detach().clone())
    x_up_kernel = nn.Parameter(x_up.detach().clone())

    torch_out = SwiGLU(x_gate_torch) * x_up_torch
    kernel_out = LlamaActCombine.apply(x_gate_kernel, x_up_kernel)
    atol = 1e-5 if dtype == torch.float32 else 5e-2
    assert torch.allclose(torch_out, kernel_out, atol=atol)

    torch_out.mean().backward()
    kernel_out.mean().backward()
    assert all(grad is not None for grad in [x_gate_torch.grad, x_up_torch.grad, x_gate_kernel.grad, x_up_kernel.grad])
    assert torch.allclose(x_gate_torch.grad, x_gate_kernel.grad, atol=atol)
    assert torch.allclose(x_up_torch.grad, x_up_kernel.grad, atol=atol)


if __name__ == '__main__':
    test_llama_act_combine(torch.float16)
