"""CPU contracts for the opt-in DiT Nemotron RMSNorm fusion (``--dit-norm-kernel``).

On Ascend NPU, ``_fused_forward`` routes contiguous bfloat16 inputs to ``torch_npu.npu_rms_norm`` inside a custom
``autograd.Function`` whose backward repeats the decomposed fp32 reference chain. These tests verify the CPU/dtype/
layout fallbacks, module patching, CLI contract, and bitwise-exact backward gradients on 2-D and 3-D inputs via a
stubbed ``torch_npu`` module.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3NemotronRMSNorm

from examples.worldmodel.cosmos3 import dit_norm_kernel as dk
from examples.worldmodel.cosmos3 import train_sft as entry


def _make_norm(dim: int, dtype: torch.dtype = torch.bfloat16) -> Cosmos3NemotronRMSNorm:
    module = Cosmos3NemotronRMSNorm(dim, eps=1e-5).to(dtype=dtype)
    with torch.no_grad():
        module.weight.copy_(torch.randn(dim, dtype=torch.float32).clamp(0.5, 1.5))
    return module


@pytest.mark.parametrize(
    "shape,dtype,contiguous,weight_dtype",
    [
        ((8, 64), torch.bfloat16, True, torch.bfloat16),
        ((8, 64), torch.float32, True, torch.float32),
        ((8, 64), torch.bfloat16, False, torch.bfloat16),
        ((8, 64), torch.bfloat16, True, torch.float32),
    ],
)
def test_fallback_matches_reference_bitwise(shape, dtype, contiguous, weight_dtype):
    torch.manual_seed(0)
    module = _make_norm(shape[-1], dtype=weight_dtype)
    base = torch.randn(shape[0], shape[1] * (1 if contiguous else 2), dtype=dtype) * 0.5
    x_raw = base[:, ::2] if not contiguous else base
    x_ref = x_raw.detach().clone().requires_grad_(True)
    x_tst = x_raw.detach().clone().requires_grad_(True)
    grad_out = torch.randn(shape, dtype=dtype) * 0.05

    ref_out = module(x_ref)
    ref_out.backward(grad_out)
    ref_gx, ref_gw = x_ref.grad.clone(), module.weight.grad.clone()

    module.weight.grad = None
    tst_out = dk._fused_forward(module, x_tst)
    tst_out.backward(grad_out)

    assert torch.equal(tst_out, ref_out)
    assert torch.equal(x_tst.grad, ref_gx)
    assert torch.equal(module.weight.grad, ref_gw)


@pytest.mark.parametrize("shape", [(16, 128), (8, 4, 128)])
@pytest.mark.parametrize("weight_trainable", [True, False])
def test_fused_backward_is_bitwise_exact_on_2d_and_3d(monkeypatch, shape, weight_trainable):
    """Both 2-D layernorms and 3-D per-head QK-norms must produce bitwise-identical gradients."""
    torch.manual_seed(7)
    module = _make_norm(shape[-1], dtype=torch.bfloat16)
    module.weight.requires_grad_(weight_trainable)
    x_base = torch.randn(*shape, dtype=torch.bfloat16) * 0.5
    grad_out = torch.randn(*shape, dtype=torch.bfloat16) * 0.05

    stub = types.ModuleType("torch_npu")
    stub.npu_rms_norm = lambda x, weight, epsilon: (dk._reference_forward(module, x),)
    monkeypatch.setitem(sys.modules, "torch_npu", stub)

    x_ref = x_base.detach().clone().requires_grad_(True)
    module.weight.grad = None
    module(x_ref).backward(grad_out)
    ref_gx = x_ref.grad.clone()
    ref_gw = module.weight.grad.clone() if weight_trainable else None

    x_tst = x_base.detach().clone().requires_grad_(True)
    module.weight.grad = None
    out = dk._FusedNemotronNorm.apply(x_tst, module.weight, float(module.eps))
    out.backward(grad_out)

    assert torch.equal(x_tst.grad, ref_gx)
    if weight_trainable:
        assert torch.equal(module.weight.grad, ref_gw)
    else:
        assert module.weight.grad is None


def test_enable_dit_norm_patches_only_nemotron_modules(monkeypatch):
    class ToyDiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = Cosmos3NemotronRMSNorm(32, eps=1e-5)
            self.n2 = Cosmos3NemotronRMSNorm(16, eps=1e-5)
            self.plain = RMSNorm(32, eps=1e-5, elementwise_affine=True, bias=False)
            self.proj = torch.nn.Linear(32, 32)

    model = ToyDiT()
    n1_orig = model.n1.forward
    plain_orig = model.plain.forward

    monkeypatch.delenv("COSMOS3_DIT_NORM_KERNEL", raising=False)
    monkeypatch.delenv("COSMOS3_DIT_NORM_VERIFY", raising=False)
    dk.enable_from_env(model)
    assert not hasattr(model, "fused_dit_norm_modules")
    assert model.n1.forward == n1_orig

    monkeypatch.setenv("COSMOS3_DIT_NORM_KERNEL", "1")
    monkeypatch.setenv("COSMOS3_DIT_NORM_VERIFY", "1")
    try:
        dk.enable_from_env(model)
        assert model.fused_dit_norm_modules == 2
        assert model.n1.forward.__func__ is dk._fused_forward
        assert model.n2.forward.__func__ is dk._fused_forward
        assert model.plain.forward == plain_orig
        assert dk._VERIFY["on"] is True
    finally:
        dk._VERIFY["on"] = False


def test_cli_flag_defaults_off_and_toggles():
    default_args = entry.parse_args(["--data", "/data", "--checkpoint", "/ckpt"])
    assert default_args.dit_norm_kernel is False

    enabled = entry.parse_args(["--data", "/data", "--checkpoint", "/ckpt", "--dit-norm-kernel"])
    assert enabled.dit_norm_kernel is True

    disabled = entry.parse_args(
        ["--data", "/data", "--checkpoint", "/ckpt", "--dit-norm-kernel", "--no-dit-norm-kernel"]
    )
    assert disabled.dit_norm_kernel is False
