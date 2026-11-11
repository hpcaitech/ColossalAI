"""Checks on the rectified-flow target and the loss that training cannot make.

Overfitting a single sample shows that the optimization path works, but it cannot show that the target is the
right one: with 1.4B trainable parameters and one sample, almost any mapping is learnable, **a flipped target sign
included**. The sign, the conditioning mask and the channel masking therefore have to be checked directly, which
needs no model at all -- only the (x_t, target) pair the packer produces and the loss applied to constructed
inputs.

Runs on CPU in a second; no checkpoint, no dataset, no device.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packing import DROID_DOMAIN_ID, compute_loss, pack_one, rectified_flow_noise

SIGMA_SCALE = 1000.0  # the forward multiplies the timestep by timestep_scale = 0.001


class _Cfg:
    """The two config fields pack_one reads."""

    latent_patch_size = 2
    action_dim = 64


@pytest.fixture(scope="module")
def packed():
    """One synthetic sample, packed. Small enough to run anywhere."""
    torch.manual_seed(0)
    latent = torch.randn(1, 4, 5, 8, 8)  # [1, C, T, H, W]
    action = torch.randn(32, 8)
    state = torch.randn(8)
    input_ids = torch.arange(7)
    generator = torch.Generator().manual_seed(1)
    inputs, targets = pack_one(latent, action, input_ids, _Cfg(), state=state, device="cpu", generator=generator)
    return latent, action, state, inputs, targets


def test_noisy_frames_satisfy_the_rectified_flow_identity(packed):
    """``x_t = (1-s) x0 + s noise`` and ``target = noise - x0`` imply ``x0 = x_t - s * target``."""
    latent, _, _, inputs, targets = packed
    sigma = inputs["vision_timesteps"][0].item() / SIGMA_SCALE
    recon = inputs["vision_tokens"][0] - sigma * targets["vision"]
    error = ((recon - latent).abs() * targets["vision_noisy_mask"]).max().item()
    assert error < 1e-5, error


def test_a_flipped_target_sign_breaks_the_identity(packed):
    """Guards the check above: it has to fail on the wrong sign, or it is not testing anything."""
    latent, _, _, inputs, targets = packed
    sigma = inputs["vision_timesteps"][0].item() / SIGMA_SCALE
    flipped = -targets["vision"]
    recon = inputs["vision_tokens"][0] - sigma * flipped
    error = ((recon - latent).abs() * targets["vision_noisy_mask"]).max().item()
    assert error > 1e-2, error


def test_the_conditioning_frame_is_clean_and_outside_the_identity(packed):
    """Frame 0 is handed to the model unnoised, so it does not satisfy the identity. That is the definition."""
    latent, _, _, inputs, targets = packed
    sigma = inputs["vision_timesteps"][0].item() / SIGMA_SCALE
    assert torch.equal(inputs["vision_tokens"][0][0, :, 0], latent[0, :, 0])
    assert not torch.equal(inputs["vision_tokens"][0][0, :, 1], latent[0, :, 1])
    recon = inputs["vision_tokens"][0] - sigma * targets["vision"]
    assert (recon - latent)[0, :, 0].abs().max().item() > 1e-2


def test_the_conditioning_frame_target_is_not_zeroed(packed):
    """It is excluded by the mask, not by zeroing: the denominator still counts those elements."""
    _, _, _, _, targets = packed
    assert targets["vision"][0, :, 0].abs().max().item() > 0
    assert targets["vision_noisy_mask"][0].sum().item() == 0
    assert targets["vision_noisy_mask"][1:].min().item() == 1


def test_the_initial_state_step_is_the_raw_state(packed):
    """``use_state=True`` prepends the state as a conditioning action step, unnoised and zero-padded."""
    _, _, state, inputs, targets = packed
    action_tokens = inputs["action_tokens"][0]
    assert torch.allclose(action_tokens[0, :8], state, atol=1e-6)
    assert action_tokens[0, 8:].abs().max().item() == 0.0
    assert targets["action_noisy_mask"][0].sum().item() == 0


def test_noisy_frame_indexes_and_loss_indexes_cover_only_noisy_frames(packed):
    """Contracts 1-3 of pack_one: the three lists have to agree on which frames are noisy."""
    _, _, _, inputs, _ = packed
    t_latent, hp, wp = inputs["vision_token_shapes"][0]
    spatial = hp * wp
    n_act = inputs["action_token_shapes"][0][0]

    assert inputs["vision_noisy_frame_indexes"][0].tolist() == list(range(1, t_latent))
    assert inputs["vision_mse_loss_indexes"].numel() == (t_latent - 1) * spatial
    assert inputs["vision_timesteps"].numel() == (t_latent - 1) * spatial
    # The input itself still covers every frame; only the loss head is restricted.
    assert inputs["vision_sequence_indexes"].numel() == t_latent * spatial

    assert inputs["action_noisy_frame_indexes"][0].tolist() == list(range(1, n_act))
    assert inputs["action_mse_loss_indexes"].numel() == n_act - 1
    assert inputs["action_timesteps"].numel() == n_act - 1
    assert inputs["action_sequence_indexes"].numel() == n_act


def _loss(targets, vision_pred, action_pred):
    batched = {
        "vision": [targets["vision"]],
        "action": [targets["action"]],
        "vision_noisy_mask": [targets["vision_noisy_mask"]],
        "action_noisy_mask": [targets["action_noisy_mask"]],
        "raw_action_dim": targets["raw_action_dim"],
    }
    out = types.SimpleNamespace(sample=[vision_pred], action=[action_pred])
    return compute_loss(out, batched).item()


def test_a_perfect_prediction_gives_zero_loss(packed):
    """Ties the loss to the target form: if the two disagreed, this would not be zero."""
    _, _, _, _, targets = packed
    assert _loss(targets, targets["vision"].clone(), targets["action"].clone()) < 1e-8


def test_the_mask_removes_the_conditioning_frame_and_the_padded_channels(packed):
    """Perturbing a masked-out element must not move the loss, while a supervised one must."""
    _, _, _, _, targets = packed
    zeros_v = torch.zeros_like(targets["vision"])
    zeros_a = torch.zeros_like(targets["action"])
    base = _loss(targets, zeros_v, zeros_a)
    assert base > 0

    conditioning = dict(targets, vision=targets["vision"].clone())
    conditioning["vision"][0, :, 0] += 1000.0
    assert abs(_loss(conditioning, zeros_v, zeros_a) - base) < 1e-6

    padded = dict(targets, action=targets["action"].clone())
    padded["action"][:, targets["raw_action_dim"] :] += 1000.0
    assert abs(_loss(padded, zeros_v, zeros_a) - base) < 1e-6

    # Control: the loss does respond to a supervised element, so the two checks above mean something.
    supervised = dict(targets, vision=targets["vision"].clone())
    supervised["vision"][0, :, 1] += 1000.0
    assert _loss(supervised, zeros_v, zeros_a) > base * 2


def test_the_action_domain_id_is_droids(packed):
    """The action projectors are per-embodiment; DROID's slot is 8.

    ``action_proj_in`` / ``action_proj_out`` are ``DomainAwareLinear``, which indexes an ``nn.Embedding`` of
    weights by ``action_domain_ids``. A wrong id reads, and trains, another embodiment's projector; DROID's own
    stays at its pretrained value. Training says nothing: the loss falls either way, because the projector it
    picked up is trainable too. Only this comparison against
    ``domain_utils.py::EMBODIMENT_TO_DOMAIN_ID["droid_lerobot"]`` catches it.
    """
    _, _, _, inputs, _ = packed
    assert DROID_DOMAIN_ID == 8
    (domain_ids,) = inputs["action_domain_ids"]
    assert domain_ids.reshape(-1).tolist() == [DROID_DOMAIN_ID]


def test_the_noise_and_the_target_are_fp32():
    """Whatever dtype the payload arrives in, the flow-matching arithmetic is fp32, as upstream's is.

    Upstream keeps the clean payload in fp32 inside its ``PackedSequence`` and noises after packing, so its
    noise, its ``x_t`` and its target are fp32. Ours used to draw the noise in ``x0.dtype`` -- bf16, the
    payload coming out of a bf16 VAE -- which quantizes the samples at about 0.4% relative and hands that to
    the target, which is what the gradient is taken against.

    The check is on a bf16 input, because an fp32 input would pass even with the old code.
    """
    x0 = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    x_t, target = rectified_flow_noise(x0, torch.tensor(0.3, dtype=torch.float32))
    assert x_t.dtype == torch.float32, f"x_t came back {x_t.dtype}, so the interpolation ran in the payload dtype"
    assert target.dtype == torch.float32, f"the target came back {target.dtype}, and the gradient is taken on it"


def test_the_whole_interpolation_is_fp32_for_a_bf16_payload():
    """Not just the dtypes: the values are the fp32 formula, element for element.

    Recomputing the same draw in fp32 and demanding exact equality is what distinguishes this from the old
    code, which drew the noise in the payload's dtype and interpolated there. A dtype check alone would pass
    on a version that computed in bf16 and cast the result up afterwards.

    What this does not cover is the sigma the caller builds: ``torch.tensor(0.3, dtype=torch.bfloat16)`` has
    already rounded 0.3 before this function sees it, and no arithmetic here can undo that -- which is why
    ``pack_one`` builds it as fp32, and why a comparison pinned at sigma 0.5, exact in bf16, could not have
    shown the problem.
    """
    x0 = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    t = torch.tensor(0.3, dtype=torch.float32)

    x_t, target = rectified_flow_noise(x0, t, torch.Generator().manual_seed(0))

    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float32)
    expected_x_t = (1.0 - t) * x0.float() + t * noise
    assert torch.equal(x_t, expected_x_t), "x_t is not the fp32 interpolation of the same draw"
    assert torch.equal(target, noise - x0.float()), "the target is not the fp32 difference of the same draw"
