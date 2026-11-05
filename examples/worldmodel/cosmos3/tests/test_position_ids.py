"""Checks on the 3D mRoPE position ids.

``position_ids`` is ``[3, sequence_length]``: the temporal, height and width axes of an interleaved mRoPE, woven
together by ``Cosmos3VLTextRotaryEmbedding.apply_interleaved_mrope``. A flat ``arange`` repeated three times is
accepted by every shape check in the forward and collapses the embedding to a 1D rope over sequence position,
which is not what the vision tower was pretrained with. Nothing in a training run reports it.

The reference below is a transcription of ``get_3d_mrope_ids_vae_tokens`` from
``data/generator/sequence_packing/mrope.py``, kept deliberately close to the original so the comparison is
against upstream's arithmetic rather than against a second copy of this module's.

Runs on CPU in a second; no checkpoint, no dataset, no device.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packing import (
    ACTION_START_FRAME_OFFSET,
    MROPE_BASE_FPS,
    MROPE_TEMPORAL_MARGIN,
    VAE_TEMPORAL_COMPRESSION,
    build_mrope_position_ids,
)

N_TEXT = 11
GRID = (5, 3, 4)  # (t_latent, hp, wp)
N_ACT = 9
FPS = 15.0


def reference_grid_ids(grid_t, grid_h, grid_w, temporal_offset, fps, tcf, base_tcf, start_frame_offset):
    """Transcribed from mrope.py::get_3d_mrope_ids_vae_tokens, fps-modulated branch."""
    tps = fps / tcf
    base_tps = MROPE_BASE_FPS / base_tcf
    frame_indices = torch.arange(grid_t, dtype=torch.float32)
    scaled_t = (frame_indices + start_frame_offset) / tps * base_tps + temporal_offset
    t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    h_index = torch.arange(grid_h).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
    return torch.stack([t_index, h_index.float(), w_index.float()], dim=0)


@pytest.fixture(scope="module")
def ids():
    return build_mrope_position_ids(N_TEXT, GRID, N_ACT, FPS)


@pytest.fixture(scope="module")
def blocks(ids):
    t_latent, hp, wp = GRID
    n_vis = t_latent * hp * wp
    return ids[:, :N_TEXT], ids[:, N_TEXT : N_TEXT + n_vis], ids[:, N_TEXT + n_vis :]


def test_shape(ids):
    t_latent, hp, wp = GRID
    assert ids.shape == (3, N_TEXT + t_latent * hp * wp + N_ACT)


def test_text_shares_one_index_across_the_three_axes(blocks):
    text, _, _ = blocks
    assert torch.equal(text[0], torch.arange(N_TEXT, dtype=text.dtype))
    assert torch.equal(text[0], text[1]) and torch.equal(text[1], text[2])


def test_vision_matches_the_upstream_grid(blocks):
    """The whole point: the vision block is a (T, H, W) grid, not three copies of one counter."""
    _, vision, _ = blocks
    t_latent, hp, wp = GRID
    expected = reference_grid_ids(
        t_latent,
        hp,
        wp,
        temporal_offset=N_TEXT + MROPE_TEMPORAL_MARGIN,
        fps=FPS,
        tcf=VAE_TEMPORAL_COMPRESSION,
        base_tcf=VAE_TEMPORAL_COMPRESSION,
        start_frame_offset=0,
    )
    assert torch.allclose(vision, expected)


def test_action_matches_the_upstream_grid(blocks):
    """Action is a (T, 1, 1) grid at frame rate, normalized against the vision base."""
    _, _, action = blocks
    expected = reference_grid_ids(
        N_ACT,
        1,
        1,
        temporal_offset=N_TEXT + MROPE_TEMPORAL_MARGIN,
        fps=FPS,
        tcf=1,  # actions are at frame rate
        base_tcf=VAE_TEMPORAL_COMPRESSION,
        start_frame_offset=ACTION_START_FRAME_OFFSET,
    )
    assert torch.allclose(action, expected)


def test_the_three_axes_are_not_all_equal(blocks):
    """The regression guard. A flat arange repeated three times passes every shape check in the forward."""
    _, vision, _ = blocks
    assert not torch.equal(vision[0], vision[1])
    assert not torch.equal(vision[1], vision[2])


def test_spatial_axes_are_a_grid_restarting_at_zero(blocks):
    """reset_spatial_indices=True: H and W restart at 0 for the segment, and repeat per frame."""
    _, vision, _ = blocks
    t_latent, hp, wp = GRID
    h, w = vision[1].view(t_latent, hp, wp), vision[2].view(t_latent, hp, wp)
    for frame in range(t_latent):
        assert torch.equal(h[frame, :, 0], torch.arange(hp, dtype=h.dtype))
        assert torch.equal(w[frame, 0, :], torch.arange(wp, dtype=w.dtype))
        assert (h[frame] == h[frame][:, :1]).all()  # constant along w
        assert (w[frame] == w[frame][:1, :]).all()  # constant along h


def test_vision_and_action_share_one_time_axis(blocks):
    """Both streams run in parallel in time, so every 4th action lands on a latent frame.

    The tolerance is loose because these are float32 values near MROPE_TEMPORAL_MARGIN, where one ulp is about
    1e-3; a difference of two of them keeps that error, not the magnitude of the difference. Upstream computes
    them the same way, in float32 with the same margin, so this is its arithmetic and not a loss introduced here.
    """
    _, vision, action = blocks
    t_latent, hp, wp = GRID
    frame_t = vision[0].view(t_latent, hp * wp)[:, 0]
    atol = 2e-3
    # 15 fps, compression 4: latent frames 1.6 apart, actions 0.4 apart.
    assert torch.allclose(frame_t[1] - frame_t[0], torch.tensor(1.6), atol=atol)
    assert torch.allclose(action[0][1] - action[0][0], torch.tensor(0.4), atol=atol)
    assert torch.allclose(action[0][VAE_TEMPORAL_COMPRESSION - 1], frame_t[1], atol=atol)


def test_the_generation_block_clears_the_text_by_the_margin(blocks):
    """The margin keeps the AR and DM blocks off each other's temporal positions."""
    text, vision, _ = blocks
    assert vision[0].min() - text[0].max() >= MROPE_TEMPORAL_MARGIN


def test_the_positions_are_fractional_and_float(ids):
    """Under fps modulation the temporal axis is not integral, so anything that copies it must stay float.

    pack_batch assembles the per-sample ids into one buffer; a ``torch.long`` buffer there would truncate 1.6 to
    1 on assignment, with the shapes still lining up and nothing raised.
    """
    assert ids.dtype.is_floating_point
    assert not torch.equal(ids[0], ids[0].floor())
