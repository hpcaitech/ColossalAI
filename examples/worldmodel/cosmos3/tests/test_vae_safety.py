"""CPU numerical/state regressions; these do not validate real CUDA latents."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusers.models.autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWan,
    WanCausalConv3d,
    WanResidualDownBlock,
    WanRMS_norm,
)
from vae_chunked import _encode_chunked, enable_chunked_encode
from vae_fast_encoder import _fast_conv_forward, _fast_down_forward, disable_fast_encoder, enable_fast_encoder
from vae_fused_norm import _fused_forward


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("magnitude", [0.0, 1e-6, 1.0, 60000.0])
@pytest.mark.parametrize("channel_first", [True, False])
def test_norm_outputs_and_gradients(dtype, magnitude, channel_first):
    torch.manual_seed(12)
    norm = WanRMS_norm(4, channel_first=channel_first, images=False, bias=True).to(dtype)
    with torch.no_grad():
        norm.gamma.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=dtype).reshape(norm.gamma.shape))
        norm.bias.fill_(0.125)
    shape = (1, 4, 1, 2, 2) if channel_first else (1, 1, 2, 2, 4)
    x = (torch.linspace(0.5, 1.0, 16).reshape(shape) * magnitude).to(dtype).requires_grad_()
    actual = _fused_forward(norm, x)
    expected = norm(x)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Use an analytical reference and a varied upstream gradient, so constant
    # normalized outputs cannot pass by merely being finite. Scale at zero to
    # keep the mathematically enormous d(normalize)/dx representable in FP16.
    upstream = torch.linspace(-0.7, 0.9, x.numel(), dtype=torch.float32).reshape(shape)
    loss_scale = 1e-9 if magnitude == 0 else (1e-3 if magnitude < 1e-4 else 1.0)
    grad = torch.autograd.grad((actual.float() * upstream * loss_scale).sum(), (x, norm.gamma, norm.bias))
    expected_grad = torch.autograd.grad((expected.float() * upstream * loss_scale).sum(), (x, norm.gamma, norm.bias))
    for got, want in zip(grad, expected_grad):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    # FP16 upstream values this small underflow before the normalize backward;
    # the representable zero-input derivative is tested separately below.
    if magnitude != 0 or dtype != torch.float16:
        assert grad[0].abs().max() > 0
        assert grad[2].abs().max() > 0
    if magnitude:
        assert grad[1].abs().max() > 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_large_values_do_not_overflow_fp32_reduction(dtype):
    norm = WanRMS_norm(4, channel_first=True, images=False, bias=True).to(dtype)
    x = (torch.tensor([1.0, -2.0, 3.0, -4.0]).reshape(1, 4, 1, 1, 1) * 1e30).to(dtype).requires_grad_()
    out = _fused_forward(norm, x)
    reference = F.normalize(x.double(), dim=1).to(dtype) * norm.scale * norm.gamma + norm.bias
    torch.testing.assert_close(out, reference, rtol=2 * torch.finfo(dtype).eps, atol=0)
    upstream = torch.tensor([0.2, -0.3, 0.7, 0.9]).reshape_as(out)
    dx = torch.autograd.grad((out.float() * upstream).sum(), x, retain_graph=True)[0]
    expected_dx = torch.autograd.grad((reference.float() * upstream).sum(), x)[0]
    assert torch.isfinite(dx).all() and dx.abs().max() > 0
    torch.testing.assert_close(dx, expected_dx, rtol=4 * torch.finfo(dtype).eps, atol=1e-37)


def test_fp16_zero_input_has_correct_representable_derivative():
    norm = WanRMS_norm(1, channel_first=True, images=False, bias=True).half()
    x = torch.zeros(1, 1, 1, 1, 1, dtype=torch.float16, requires_grad=True)
    out = _fused_forward(norm, x)
    # The smallest FP16 subnormal as upstream yields a finite nonzero derivative:
    # d(x / eps)/dx = 1e12, so expected dx is about 59604.6, rounded to FP16.
    upstream = torch.full_like(out, 2.0**-24)
    dx, dbias = torch.autograd.grad(out, (x, norm.bias), grad_outputs=upstream)
    expected = (upstream.float() / 1e-12).half()
    torch.testing.assert_close(dx, expected, rtol=0, atol=0)
    assert torch.isfinite(dx).all() and (dx > 0).all()
    assert (dbias > 0).all()


def test_near_zero_fp16_is_normalized_to_one():
    norm = WanRMS_norm(4, channel_first=True, images=False).half()
    x = torch.full((1, 4, 1, 1, 1), 1e-6, dtype=torch.float16)
    torch.testing.assert_close(_fused_forward(norm, x), torch.ones_like(x), rtol=0, atol=0)


def test_scalar_nonzero_bias_is_not_discarded():
    norm = WanRMS_norm(4, channel_first=True, images=False)
    norm.bias = 0.75
    x = torch.zeros(1, 4, 1, 1, 1)
    torch.testing.assert_close(_fused_forward(norm, x), torch.full_like(x, 0.75))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cache_frames", [0, 1, 2, 4])
def test_fast_conv_prime_tail_and_cache_oversize(dtype, cache_frames):
    torch.manual_seed(4)
    conv = WanCausalConv3d(2, 3, 3, padding=1).to(dtype).eval()
    for frames in (1, 4, 8):
        x = torch.randn(1, 2, frames, 4, 5).to(dtype)
        cache = torch.randn(1, 2, cache_frames, 4, 5).to(dtype) if cache_frames else None
        saved_x = x.clone()
        saved_cache = None if cache is None else cache.clone()
        with torch.no_grad():
            torch.testing.assert_close(_fast_conv_forward(conv, x, cache), conv(x, cache), rtol=0, atol=0)
        assert torch.equal(x, saved_x)
        if cache is not None:
            assert torch.equal(cache, saved_cache)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fast_downblock_cached_sequence_and_new_clip(dtype):
    torch.manual_seed(7)
    block = WanResidualDownBlock(2, 4, 0.0, num_res_blocks=1, temperal_downsample=True, down_flag=True).to(dtype).eval()
    for _ in range(2):
        original_cache, fast_cache = [None] * 32, [None] * 32
        for frames in (1, 8, 4):
            x = torch.randn(1, 2, frames, 4, 4).to(dtype)
            with torch.no_grad():
                expected = block(x, feat_cache=original_cache, feat_idx=[0])
                actual = _fast_down_forward(block, x, fast_cache, [0])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for got, want in zip(fast_cache, original_cache):
                if want is None:
                    assert got is None
                else:
                    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("residual", [False, True])
def test_tiny_real_cpu_vae_chunks_fast_and_repeated_clips(dtype, residual):
    # Random tiny CPU model: integration coverage for cache semantics only;
    # this does not qualify a production checkpoint or GPU kernel.
    torch.manual_seed(11)
    vae = (
        AutoencoderKLWan(
            base_dim=4,
            decoder_base_dim=4,
            z_dim=2,
            dim_mult=[1, 2, 2, 2],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
            is_residual=residual,
        )
        .eval()
        .to(dtype)
    )
    for frames in (1, 14, 13):
        x = torch.randn(1, 3, frames, 8, 8).to(dtype)
        with torch.no_grad():
            reference = vae._encode(x)
            wider = _encode_chunked(vae, x, 8)
            enable_fast_encoder(vae)
            fast = _encode_chunked(vae, x, 8)
            disable_fast_encoder(vae)
        # Grouping convolutions may change backend accumulation order.
        torch.testing.assert_close(wider, reference, rtol=2 * torch.finfo(dtype).eps, atol=2e-7)
        torch.testing.assert_close(fast, wider, rtol=2 * torch.finfo(dtype).eps, atol=2e-7)
        assert all(entry is None for entry in vae._enc_feat_map)


class _StatefulEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fail = False

    def forward(self, x, feat_cache, feat_idx):
        previous = feat_cache[0]
        if previous is None:
            assert x.shape[2] == 1
            out = x.clone()
        else:
            # A causal four-frame mean uses all frames of each group. This
            # catches state/phase bugs that an every-fourth-frame stub misses.
            out = x.reshape(*x.shape[:2], -1, 4, *x.shape[3:]).mean(dim=3) + previous
        feat_cache[0] = out[:, :, -1:].clone()
        if self.fail:
            raise RuntimeError("injected encoder failure")
        return out


class _VAE:
    def __init__(self):
        self.config = SimpleNamespace(patch_size=None)
        self.use_tiling = False
        self.encoder = _StatefulEncoder()
        self.quant_conv = torch.nn.Identity()
        self.clear_cache()

    def clear_cache(self):
        self._enc_feat_map = [None]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("frames", [1, 5, 13, 14, 33])
def test_chunk_prime_tail_independent_calls(dtype, frames):
    # Stateful toy is scheduling coverage, not a real VAE latent gate.
    vae = _VAE()
    video = torch.ones(1, 2, frames, 2, 2, dtype=dtype)

    # Compare grouped calls with a phase-correct per-group stateful reference.
    # A chunk's cache should reflect all groups, not just the call count.
    def run(chunk, cache):
        if cache[0] is None:
            return vae.encoder(chunk, cache, [0])
        parts = [vae.encoder(part, cache, [0]) for part in chunk.split(4, dim=2)]
        return torch.cat(parts, dim=2)

    for magnitude in (1.0, 3.0):
        reference = _encode_chunked(vae, video * magnitude, 4, run)
        actual = _encode_chunked(vae, video * magnitude, 12, run)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert all(entry is None for entry in vae._enc_feat_map)
        assert actual.shape[2] == 1 + (frames - 1) // 4


def test_chunk_cache_cleared_after_failure_and_tiling():
    vae = _VAE()
    vae.encoder.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        _encode_chunked(vae, torch.ones(1, 2, 5, 2, 2), 4)
    assert vae._enc_feat_map == [None]
    vae.use_tiling = True
    vae.tile_sample_min_width = vae.tile_sample_min_height = 1

    def tiled(x):
        vae._enc_feat_map[0] = x
        return x.clone()

    vae.tiled_encode = tiled
    _encode_chunked(vae, torch.ones(1, 2, 5, 2, 2), 4)
    assert vae._enc_feat_map == [None]


@pytest.mark.parametrize("chunk", [-4, 0, 3, 4.5])
def test_invalid_chunk_rejected(chunk):
    with pytest.raises(ValueError, match="positive multiple"):
        _encode_chunked(_VAE(), torch.ones(1, 2, 5, 2, 2), chunk)


def test_empty_clip_and_aot_without_chunks_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        _encode_chunked(_VAE(), torch.empty(1, 2, 0, 2, 2), 4)
    with pytest.raises(ValueError, match="positive multiple"):
        enable_chunked_encode(_VAE(), 0, aot_dir="unused")
