"""Checks on the layout of the packed sequence, against upstream's own arithmetic.

These three came out of comparing our packed input with ``PackedSequence`` on one real window (episode 26827,
offset 0), where upstream's object and our kwargs are the same contract seen from two sides. Each of the three
produced a sequence that was internally consistent and silently different from upstream's:

1. **The canvas padding reached the latent.** ``resize_and_reflect_pad`` fits the 540x640 composite inside a
   544x736 bucket canvas and reflection-pads it. Upstream encodes the padded frame and then drops the padding
   in latent space (``omni_mot_model.py::_remove_padding_from_latent``). We kept it, so the vision stream
   carried 34x46 latent cells instead of 33x40 -- 3519 tokens instead of 3060, and 13% of the vision loss
   computed on reflected pixels.
2. **The patch grid floored an odd latent side.** The forward pads up (``_patchify_and_pack_latents`` computes
   ``ceil(h / p) * p``), and so does upstream's packer; flooring reserved 16 rows of sequence for the 17 the
   forward produces. Unreachable until (1) was fixed, because 544/16 = 34 is even.
3. **The text span was missing its last two tokens.** Upstream's span is
   ``caption + EOS + start_of_generation`` (``PackedSequenceBuilder.pack_text_tokens``), and without them every
   index after the text sat two places early.

Runs on CPU with a stub VAE; no checkpoint, no dataset, no device.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packing import START_OF_GENERATION, close_text_span, pack_batch

LATENT_CHANNELS = 48
SPATIAL = 16  # the Wan 2.2 VAE's spatial compression factor
PATCH = 2
ACTION_DIM = 64
CHUNK = 32
FRAMES = 33
CANVAS_H, CANVAS_W = 544, 736
CONTENT_H, CONTENT_W = 540, 640  # DROID's concat_view composite inside that canvas


class StubVAE:
    """Encodes to a zero latent of the right shape, with the channel statistics normalize_latent reads.

    The values do not matter here -- the layout does -- but the shapes must follow the real VAE's compression,
    because the crop is derived from the ratio between the video and the latent.
    """

    def __init__(self):
        self.config = SimpleNamespace(
            latents_mean=[0.0] * LATENT_CHANNELS,
            latents_std=[1.0] * LATENT_CHANNELS,
            z_dim=LATENT_CHANNELS,
        )
        self._p = torch.zeros(1, dtype=torch.float32)

    def parameters(self):
        return iter([self._p])

    def encode(self, video):
        n, _, t, h, w = video.shape
        t_latent = (t - 1) // 4 + 1
        return torch.zeros(n, LATENT_CHANNELS, t_latent, h // SPATIAL, w // SPATIAL, dtype=torch.float32)


class StubTokenizer:
    """Returns a fixed caption, and knows the two special tokens the text span ends with."""

    eos_token_id = 151645
    unk_token_id = None
    N_CAPTION = 11

    def apply_chat_template(self, *_a, **_kw):
        return list(range(100, 100 + self.N_CAPTION))

    def convert_tokens_to_ids(self, token):
        return {START_OF_GENERATION: 151652}.get(token)


class StubPolicy:
    def __init__(self):
        self.meta = None

    def set_packed_meta(self, meta):
        self.meta = meta


def make_batch():
    return {
        "video": torch.zeros(1, 3, FRAMES, CANVAS_H, CANVAS_W, dtype=torch.float32),
        "action": torch.zeros(1, CHUNK, 8),
        "state": torch.zeros(1, 8),
        "task": ["wipe the tabletop"],
        "canvas_hw": torch.tensor([[CANVAS_H, CANVAS_W]]),
        "content_hw": torch.tensor([[CONTENT_H, CONTENT_W]]),
        "aspect_key": ["4,3"],
        "idle_frames": [3],
        "additional_view_description": ["three views"],
    }


@pytest.fixture(scope="module")
def packed():
    policy = StubPolicy()
    inputs, targets = pack_batch(
        make_batch(),
        StubVAE(),
        StubTokenizer(),
        SimpleNamespace(latent_patch_size=PATCH, action_dim=ACTION_DIM),
        policy=policy,
        num_frames=FRAMES,
        fps=15.0,
        resolution="480",
        device="cpu",
        generator=torch.Generator().manual_seed(0),
    )
    return inputs, targets, policy


def test_the_canvas_padding_does_not_reach_the_latent(packed):
    inputs, _, _ = packed
    latent = inputs["vision_tokens"][0]
    assert latent.shape[-2:] == (CONTENT_H // SPATIAL, CONTENT_W // SPATIAL) == (33, 40)
    # And not the canvas, which is what we used to pack.
    assert latent.shape[-2:] != (CANVAS_H // SPATIAL, CANVAS_W // SPATIAL)


def test_an_odd_latent_side_is_padded_up_not_truncated(packed):
    inputs, _, _ = packed
    t_latent = (FRAMES - 1) // 4 + 1
    expected = (t_latent, -(-33 // PATCH), -(-40 // PATCH))
    assert tuple(inputs["vision_token_shapes"][0]) == expected == (9, 17, 20)
    # The sequence has to reserve exactly what the forward's own patchify will produce.
    assert inputs["vision_sequence_indexes"].numel() == 9 * 17 * 20


def test_the_text_span_ends_with_eos_then_start_of_generation(packed):
    inputs, _, _ = packed
    ids = inputs["input_ids"].tolist()
    assert len(ids) == StubTokenizer.N_CAPTION + 2
    assert ids[-2:] == [StubTokenizer.eos_token_id, 151652]
    assert inputs["und_len"] == len(ids)


def test_the_sequence_length_is_the_sum_of_the_three_blocks(packed):
    inputs, _, policy = packed
    n_text = inputs["und_len"]
    n_vis = inputs["vision_sequence_indexes"].numel()
    n_act = inputs["action_sequence_indexes"].numel()
    assert (n_text, n_vis, n_act) == (StubTokenizer.N_CAPTION + 2, 9 * 17 * 20, CHUNK + 1)
    assert inputs["sequence_length"] == n_text + n_vis + n_act
    assert inputs["position_ids"].shape == (3, inputs["sequence_length"])
    # On the real window the caption is 149 tokens, which makes this same sum upstream's 3244.
    assert 149 + 2 + n_vis + n_act == 3244
    assert list(policy.meta.ar_lens) == [n_text]
    assert list(policy.meta.dm_lens) == [n_vis + n_act]


def test_close_text_span_refuses_a_tokenizer_without_the_generation_marker():
    class NoMarker(StubTokenizer):
        def convert_tokens_to_ids(self, token):
            return None

    with pytest.raises(ValueError, match="generated block"):
        close_text_span([1, 2, 3], NoMarker())


def test_a_vae_with_unequal_spatial_compression_is_refused():
    class Lopsided(StubVAE):
        def encode(self, video):
            n, _, t, h, w = video.shape
            return torch.zeros(n, LATENT_CHANNELS, (t - 1) // 4 + 1, h // SPATIAL, w // (SPATIAL * 2))

    with pytest.raises(ValueError, match="equally"):
        pack_batch(
            make_batch(),
            Lopsided(),
            StubTokenizer(),
            SimpleNamespace(latent_patch_size=PATCH, action_dim=ACTION_DIM),
            policy=StubPolicy(),
            num_frames=FRAMES,
            fps=15.0,
            resolution="480",
            device="cpu",
            generator=torch.Generator().manual_seed(0),
        )


def test_the_timesteps_are_fp32_whatever_the_payload_dtype_is(packed):
    """The forward projects the timestep before casting, and upstream carries it in fp32.

    In bf16 a sampled timestep of 734.21 is stored as 734.0. The pinned sigma of 0.5 gives 500.0, which is
    exact in bf16, so a comparison run at that sigma cannot show this.
    """
    inputs, _, _ = packed
    assert inputs["vision_timesteps"].dtype is torch.float32
    assert inputs["action_timesteps"].dtype is torch.float32
    # x_t leaves in the dtype the model computes in, which pack_batch takes as payload_dtype and which
    # defaults to bf16. x0 itself is fp32 all the way to the interpolation, as upstream keeps it.
    assert inputs["vision_tokens"][0].dtype is torch.bfloat16


def test_the_action_padding_carries_no_noise(packed):
    """Upstream noises the full padded width and then clears the padding.

    ``omni_mot_model.py::_add_noise_to_input`` ends with
    ``xt_action[i][:, raw_action_dim:] = 0``. Those 56 channels are not inert: ``action_proj_in`` is a
    64 -> 4096 projection, so noise left there is projected into the hidden state. Measured on episode 26827
    offset 0 at a pinned sigma of 0.5, leaving it in gave our action rows 52% more norm than upstream's at the
    input of block 0, while the never-noised conditioning row matched bit for bit.
    """
    inputs, targets, _ = packed
    tokens = inputs["action_tokens"][0]
    dim = targets["raw_action_dim"]
    assert dim == 8 and tokens.shape[-1] == ACTION_DIM
    assert torch.count_nonzero(tokens[:, dim:]) == 0
    # The real channels are still noised: row 0 is the clean conditioning action, the rest are not.
    assert not torch.equal(tokens[1:, :dim], torch.zeros_like(tokens[1:, :dim]))
