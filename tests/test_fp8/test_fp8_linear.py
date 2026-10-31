import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from colossalai.accelerator import get_accelerator
from colossalai.quantization.fp8 import _LinearFp8, linear_fp8
from colossalai.utils import get_current_device

D_IN, D_OUT = 16, 32
B, S = 2, 64


def check_fp8_linear(linear, use_bias: bool, use_batch: bool, dtype: torch.dtype):
    torch.manual_seed(42)
    # create tensors
    w = torch.rand(D_OUT, D_IN, device=get_current_device(), dtype=dtype, requires_grad=True)
    ref_w = w.clone().detach().requires_grad_()
    if use_batch:
        x_shape = (B, S, D_IN)
    else:
        x_shape = (S, D_IN)
    x = torch.rand(x_shape, device=get_current_device(), dtype=dtype, requires_grad=True)
    ref_x = x.clone().detach().requires_grad_()
    if use_bias:
        bias = torch.rand(D_OUT, device=get_current_device(), dtype=dtype, requires_grad=True)
        ref_bias = bias.clone().detach().requires_grad_()
    else:
        bias = None
        ref_bias = None

    out = linear(x, w, bias)
    assert out.shape == x_shape[:-1] + (D_OUT,)
    out.sum().backward()
    ref_out = F.linear(ref_x, ref_w, ref_bias)
    ref_out.sum().backward()

    assert_close(out, ref_out, rtol=0.2, atol=0.1)
    assert_close(x.grad, ref_x.grad, rtol=0.2, atol=0.1)
    assert_close(w.grad, ref_w.grad, rtol=0.2, atol=0.1)
    if use_bias:
        assert_close(bias.grad, ref_bias.grad, rtol=0.2, atol=0.1)


@pytest.mark.skipif(get_accelerator().get_device_capability()[0] < 9, reason="Test requires device capability >= 9.0")
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("use_batch", [True, False])
def test_fp8_linear(use_bias: bool, use_batch: bool):
    # Test the public entry point, including the existing 2.13 eager fallback.
    check_fp8_linear(linear_fp8, use_bias, use_batch, torch.bfloat16)


@pytest.mark.skipif(get_accelerator().get_device_capability()[0] < 9, reason="Test requires device capability >= 9.0")
@pytest.mark.parametrize("legacy_return", [False, True], ids=["tensor-return", "tuple-return"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("use_batch", [False, True])
def test_fp8_linear_scaled_mm_return(monkeypatch, legacy_return, dtype, use_bias, use_batch):
    scaled_mm = torch._scaled_mm
    calls = []

    def scaled_mm_with_return_contract(*args, **kwargs):
        result = scaled_mm(*args, **kwargs)
        output = result[0] if isinstance(result, tuple) else result
        calls.append(output.shape)
        # amax is unused by _LinearFp8; exercise both historical return contracts.
        return (output, output.abs().max()) if legacy_return else output

    monkeypatch.setattr(torch, "_scaled_mm", scaled_mm_with_return_contract)
    # Test the actual forward and backward kernels without compiler caching of
    # the patched function. This does not claim torch.compile coverage on 2.13.
    check_fp8_linear(_LinearFp8.apply, use_bias, use_batch, dtype)
    assert len(calls) == 3, "Expected forward, input-gradient and weight-gradient matmuls"
