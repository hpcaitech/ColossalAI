"""Compare the opt-in RMSNorm forward against diffusers on CPU.

The normal range preserves FP32 normalization followed by the input-dtype cast and affine arithmetic.
Extreme values and gradient behavior have additional coverage in test_vae_safety.py. These tests do not
establish correctness or speed for production GPU/NPU checkpoints.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm
from vae_fused_norm import _fused_forward

SHAPES = [(2, 16, 3, 8, 8), (1, 8, 1, 4, 4)]


def relative(a, b):
    return (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_diffusers_in_fp32(shape):
    torch.manual_seed(0)
    norm = WanRMS_norm(shape[1], channel_first=True, images=False).eval()
    with torch.no_grad():
        norm.gamma.copy_(torch.randn_like(norm.gamma) * 0.3 + 1.0)  # the checkpoint's gamma is not all ones
    x = torch.randn(*shape)
    assert relative(_fused_forward(norm, x), norm(x)) < 1e-6


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_diffusers_in_fp16(shape):
    torch.manual_seed(0)
    norm = WanRMS_norm(shape[1], channel_first=True, images=False).eval().half()
    with torch.no_grad():
        norm.gamma.copy_(torch.randn_like(norm.gamma) * 0.3 + 1.0)
    x = torch.randn(*shape).half()
    difference = relative(_fused_forward(norm, x), norm(x))
    # Compare to the same-dtype reference; latent quantization is not a correctness criterion.
    assert difference < 2e-3, difference


def test_bias_and_image_layout():
    """A 4-D input uses dim=-1 and a (dim,) gamma; the fused forward has to follow the same convention."""
    torch.manual_seed(0)
    norm = WanRMS_norm(8, channel_first=False, images=True, bias=True).eval()
    with torch.no_grad():
        norm.gamma.copy_(torch.randn_like(norm.gamma) * 0.2 + 1.0)
        norm.bias.copy_(torch.randn_like(norm.bias) * 0.1)
    x = torch.randn(3, 5, 4, 8)
    assert relative(_fused_forward(norm, x), norm(x)) < 1e-6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_a_zero_input_does_not_blow_up(dtype):
    """A zero window must stay zero without an overflowing FP16 reciprocal."""
    norm = WanRMS_norm(4, channel_first=True, images=False).eval().to(dtype)
    out = _fused_forward(norm, torch.zeros(1, 4, 2, 2, 2, dtype=dtype))
    assert torch.isfinite(out).all()
    assert torch.equal(out, norm(torch.zeros(1, 4, 2, 2, 2, dtype=dtype)))
