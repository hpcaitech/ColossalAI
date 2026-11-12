"""The chunked encoder call must pick exactly the frames diffusers' own loop picks.

The window width is a scheduling knob, not a semantic one: the encoder carries its causal cache across calls, so
any decomposition of the clip into whole 4-frame groups has to reproduce the same latent. This checks that
property on a stub encoder that compresses 4:1 in time -- no weights, no device -- which pins the three places
the real implementation could go wrong: where a window starts, where the one-frame prime sits, and how a clip
whose frame count is not a multiple of 4 is padded and cropped back.

Runs on CPU.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vae_chunked import _encode_chunked, enable_chunked_encode


class _Compress4(torch.nn.Module):
    """Stands in for the encoder: records the window it was handed and returns every fourth frame."""

    def __init__(self):
        super().__init__()
        self.windows = []

    def forward(self, x, feat_cache=None, feat_idx=None):
        self.windows.append(x.shape[2])
        return x[:, :, ::4].clone()


class _StubVAE:
    """The attributes ``_encode_chunked`` touches, and nothing else."""

    def __init__(self):
        self.config = SimpleNamespace(patch_size=None)
        self.use_tiling = False
        self.encoder = _Compress4()
        self.quant_conv = torch.nn.Identity()
        self.clear_cache()

    def clear_cache(self):
        self._enc_feat_map = [None] * 4


def test_windows_cover_the_clip_the_way_the_encoder_expects():
    vae = _StubVAE()
    video = torch.arange(3 * 33 * 2 * 2, dtype=torch.float32).reshape(1, 3, 33, 2, 2)
    out = _encode_chunked(vae, video, chunk_frames=12)

    # One prime frame, then 12 + 12 + 8: the tail is short here, but still a whole number of 4-frame groups.
    assert vae.encoder.windows == [1, 12, 12, 8]
    # Every group of four pixel frames becomes one latent frame, and the prime frame is the first of them.
    assert out.shape[2] == 9
    assert torch.equal(out, video[:, :, [0, 1, 5, 9, 13, 17, 21, 25, 29]])


@pytest.mark.parametrize("chunk", [8, 16, 24, 32])
def test_every_window_width_gives_the_same_latent(chunk):
    video = torch.randn(1, 3, 33, 2, 2)
    reference = _encode_chunked(_StubVAE(), video, chunk_frames=4)
    assert torch.equal(_encode_chunked(_StubVAE(), video, chunk_frames=chunk), reference)


def test_a_ragged_clip_is_padded_and_cropped_back():
    vae = _StubVAE()
    video = torch.randn(1, 3, 34, 2, 2)  # 33 frames after the prime: one group short
    out = _encode_chunked(vae, video, chunk_frames=12)

    assert vae.encoder.windows == [1, 12, 12, 12]  # 36 frames, padded by three repeated frames
    assert out.shape[2] == 9  # (34 - 1) // 4 + 1, the padded tail cropped away


def test_the_window_has_to_be_a_whole_number_of_latent_chunks():
    with pytest.raises(ValueError, match="multiple of 4"):
        _encode_chunked(_StubVAE(), torch.randn(1, 3, 33, 2, 2), chunk_frames=6)


def test_enable_zero_leaves_the_vae_alone():
    vae = _StubVAE()
    assert enable_chunked_encode(vae, 0) is vae
    assert not hasattr(vae, "encode_chunk_frames")
