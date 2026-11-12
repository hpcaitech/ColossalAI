"""Both fast-encoder changes must reproduce diffusers' results bit for bit, caches included.

The encoder always runs with a feature cache and always primes it with a single frame, so the tests replay that
sequence: one T=1 call with an empty cache, then full chunks. The block configurations are the ones the Wan VAE
actually builds -- temporally downsampling with a spatial downsample, without one, and the plain last block.
(diffusers' block cannot halve time on a call that carries no cache, which is why the no-cache test only covers
blocks whose ``factor_t`` is 1.)

Runs on CPU.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanResidualDownBlock
from vae_fast_encoder import _fast_conv_forward, _fast_down_forward, disable_fast_encoder, enable_fast_encoder

CONV_CONFIGS = [
    # kernel, stride, padding, input T, cache frames
    ((3, 3, 3), 1, 1, 4, 2),
    ((3, 3, 3), 1, 1, 1, 2),
    ((3, 1, 1), 1, (1, 0, 0), 3, 2),
    ((3, 1, 1), (2, 1, 1), (0, 0, 0), 5, 1),
    ((1, 1, 1), 1, 0, 3, 0),
]

BLOCK_CONFIGS = [
    # in_dim, out_dim, temperal_downsample, down_flag, chunk T
    (4, 8, False, True, 4),
    (4, 8, True, True, 4),
    (4, 4, False, False, 4),
]


@pytest.mark.parametrize("config", CONV_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_conv_matches_diffusers(config, dtype):
    kernel, stride, padding, num_frame, cache_frames = config
    torch.manual_seed(0)
    conv = WanCausalConv3d(8, 12, kernel, stride=stride, padding=padding).eval().to(dtype)
    x = torch.randn(2, 8, num_frame, 7, 9, dtype=dtype)
    cache = torch.randn(2, 8, cache_frames, 7, 9, dtype=dtype) if cache_frames else None
    with torch.no_grad():
        expected = conv(x, cache)
        actual = _fast_conv_forward(conv, x, cache)
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), (actual - expected).abs().max().item()


@pytest.mark.parametrize("config", CONV_CONFIGS[:4])
def test_conv_with_a_one_frame_cache_still_pads_t(config):
    """A one-frame cache leaves one frame of causal pad; the fast path must still add it."""
    kernel, stride, padding, num_frame, _ = config
    torch.manual_seed(1)
    conv = WanCausalConv3d(8, 12, kernel, stride=stride, padding=padding).eval()
    x = torch.randn(1, 8, num_frame, 5, 5)
    cache = torch.randn(1, 8, 1, 5, 5)
    with torch.no_grad():
        assert torch.equal(_fast_conv_forward(conv, x, cache), conv(x, cache))


def test_conv_without_padding_never_calls_pad():
    conv = WanCausalConv3d(4, 4, 1).eval()
    x = torch.randn(1, 4, 3, 6, 6)
    with torch.no_grad():
        assert torch.equal(_fast_conv_forward(conv, x), F.conv3d(x, conv.weight, conv.bias))


def make_block(config):
    in_dim, out_dim, temperal, down, _ = config
    torch.manual_seed(0)
    return WanResidualDownBlock(
        in_dim, out_dim, 0.0, num_res_blocks=2, temperal_downsample=temperal, down_flag=down
    ).eval()


def run_block(block, chunks, fast):
    """Replay the encoder's sequence through one shared feature cache."""
    caches = [None] * 64
    outputs = []
    for chunk in chunks:
        with torch.no_grad():
            output = (
                _fast_down_forward(block, chunk, caches, [0]) if fast else block(chunk, feat_cache=caches, feat_idx=[0])
            )
        outputs.append(output)
    return outputs, caches


@pytest.mark.parametrize("config", BLOCK_CONFIGS)
def test_block_same_output_and_cache(config):
    _, _, _, _, chunk_t = config
    block = make_block(config)
    torch.manual_seed(1)
    chunks = [torch.randn(1, 4, 1, 8, 8), torch.randn(1, 4, chunk_t, 8, 8), torch.randn(1, 4, chunk_t, 8, 8)]

    expected, expected_cache = run_block(block, chunks, fast=False)
    fast, fast_cache = run_block(block, chunks, fast=True)

    for got, want in zip(fast, expected):
        assert torch.equal(got, want), (got - want).abs().max().item()
    for got, want in zip(fast_cache, expected_cache):
        if want is None:
            assert got is None
        else:
            assert torch.equal(got, want)


@pytest.mark.parametrize("config", [BLOCK_CONFIGS[0], BLOCK_CONFIGS[2]])
def test_block_without_cache(config):
    """Only blocks whose ``factor_t`` is 1 have a no-cache path that diffusers itself can run."""
    _, _, _, _, chunk_t = config
    block = make_block(config)
    x = torch.randn(1, 4, chunk_t, 8, 8)
    with torch.no_grad():
        expected = block(x)
        actual = _fast_down_forward(block, x)
    assert torch.equal(actual, expected)


def test_enable_and_disable_round_trip():
    module = torch.nn.Sequential(
        WanCausalConv3d(4, 4, 3, padding=1),
        WanResidualDownBlock(4, 4, 0.0, num_res_blocks=1, temperal_downsample=False, down_flag=False),
    )
    enable_fast_encoder(module)
    assert module[0].forward.__func__ is _fast_conv_forward
    assert module[1].forward.__func__ is _fast_down_forward
    disable_fast_encoder(module)
    assert "forward" not in module[0].__dict__
    assert "forward" not in module[1].__dict__
