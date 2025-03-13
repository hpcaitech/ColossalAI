import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from colossalai.accelerator import get_accelerator
from colossalai.quantization.fp8 import linear_fp8_deep_gemm
from colossalai.utils import get_current_device

m, k, n = 128, 384, 256
DTYPE = torch.bfloat16


@pytest.mark.skipif(get_accelerator().get_device_capability()[0] < 9, reason="Test requires device capability >= 9.0")
def test_fp8_linear():
    # create tensors
    x = torch.rand((m, k), device=get_current_device(), dtype=DTYPE, requires_grad=True)
    w = torch.rand((n, k), device=get_current_device(), dtype=DTYPE, requires_grad=True)
    bias = torch.rand(n, device=get_current_device(), dtype=DTYPE, requires_grad=True)
    ref_w = w.clone().detach().requires_grad_()
    ref_x = x.clone().detach().requires_grad_()

    out = linear_fp8_deep_gemm(x, w, bias)
    assert out.shape == x.shape[:-1] + (n,)
    out.sum().backward()
    ref_out = F.linear(ref_x, ref_w, bias)
    ref_out.sum().backward()

    assert_close(out, ref_out)
    assert_close(x.grad, ref_x.grad)
    assert_close(w.grad, ref_w.grad)
