"""Microbatch changes encode scheduling, not packed payloads or the objective.

CPU StubVAE only: these tests do not certify production VAE numerical agreement
across GPU batch sizes. Nontrivial channel statistics and odd spatial dimensions
exercise normalization, crop, ceil patch-grid metadata, loss and backward.
"""

import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packing import START_OF_GENERATION, compute_loss, normalize_latent, pack_batch


class StubVAE:
    def __init__(self, dtype, output_kind):
        self.config = SimpleNamespace(latents_mean=[-0.3, 0.1, 0.5, 0.9], latents_std=[0.7, 1.1, 1.7, 2.3], z_dim=4)
        self.parameter = torch.zeros(1, dtype=dtype)
        self.calls = []
        self.output_kind = output_kind

    def parameters(self):
        return iter([self.parameter])

    def encode(self, video):
        assert video.dtype == self.parameter.dtype
        self.calls.append(video.shape[0])
        # Per-sample pixels distinguish samples even when a microbatch starts at
        # a different index. Both axes compress by 2; time follows Wan's 4:1.
        base = video.mean(dim=1, keepdim=True)[:, :, ::4, ::2, ::2]
        channel = torch.arange(4, dtype=video.dtype).reshape(1, 4, 1, 1, 1)
        latent = base * (channel + 1) + channel * 0.2
        if self.output_kind == "tuple":
            return (latent,)
        if self.output_kind == "distribution":
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))
        return latent


class StubTokenizer:
    eos_token_id = 151645
    unk_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        count = len(messages[0]["content"]) % 7 + 2
        return list(range(100, 100 + count))

    def convert_tokens_to_ids(self, token):
        return 151652 if token == START_OF_GENERATION else None


class StubPolicy:
    def set_packed_meta(self, meta):
        self.meta = meta


def batch():
    generator = torch.Generator().manual_seed(22)
    return {
        "video": torch.randn(3, 3, 9, 12, 16, generator=generator),
        "action": torch.randn(3, 8, 3, generator=generator),
        "state": torch.randn(3, 3, generator=generator),
        "task": ["open", "move the small cup", "close the drawer carefully"],
        "canvas_hw": torch.tensor([[12, 16]] * 3),
        "content_hw": torch.tensor([[10, 14]] * 3),
        "aspect_key": ["4,3"] * 3,
        "idle_frames": [0, 1, 2],
        "additional_view_description": ["front", "front and wrist", "wrist"],
    }


def exact(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            exact(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            exact(a, b)
    else:
        assert actual == expected


def loss_and_gradient(inputs, targets):
    parameter = torch.tensor([0.7, -0.2, 1.1, 0.3], requires_grad=True)
    output = SimpleNamespace(
        sample=[value.float() * parameter[0] + parameter[1] for value in inputs["vision_tokens"]],
        action=[value.float() * parameter[2] + parameter[3] for value in inputs["action_tokens"]],
    )
    loss, components = compute_loss(output, targets, return_components=True)
    gradient = torch.autograd.grad(loss, parameter)[0]
    assert torch.isfinite(loss) and loss > 0
    assert torch.isfinite(gradient).all() and (gradient != 0).all()
    return loss.detach(), components, gradient


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("output_kind", ["tensor", "tuple", "distribution"])
def test_microbatch_preserves_inputs_fp32_targets_meta_loss_and_gradients(dtype, output_kind):
    data = batch()
    reference = None
    for size, expected_calls in ((0, [3]), (1, [1, 1, 1]), (2, [2, 1])):
        vae = StubVAE(dtype, output_kind)
        policy = StubPolicy()
        generator = torch.Generator().manual_seed(12)
        inputs, targets = pack_batch(
            data,
            vae,
            StubTokenizer(),
            SimpleNamespace(latent_patch_size=2, action_dim=8),
            policy=policy,
            num_frames=9,
            fps=15.0,
            resolution="480",
            device="cpu",
            generator=generator,
            vae_encode_batch=size,
        )
        assert vae.calls == expected_calls
        assert all(value.dtype == torch.float32 for name in ("vision", "action") for value in targets[name])
        assert all(value.dtype == torch.bfloat16 for value in inputs["vision_tokens"])
        assert all(tuple(value.shape[-2:]) == (5, 7) for value in inputs["vision_tokens"])
        assert all(tuple(shape) == (3, 3, 4) for shape in inputs["vision_token_shapes"])
        # Clean first frame gives an independent normalization/crop check, so
        # equivalence cannot pass by accidentally dropping channel statistics.
        clean_vae = StubVAE(dtype, "tensor")
        raw = clean_vae.encode(data["video"].to(dtype))
        clean = normalize_latent(raw, clean_vae).float()
        for index, value in enumerate(inputs["vision_tokens"]):
            torch.testing.assert_close(
                value[:, :, 0], clean[index : index + 1, :, 0, :5, :7].bfloat16(), rtol=0, atol=0
            )
        result = (inputs, targets, asdict(policy.meta), generator.get_state(), loss_and_gradient(inputs, targets))
        if reference is None:
            reference = result
        else:
            exact(result, reference)


def test_microbatch_invalid_size_is_rejected_before_encode():
    vae = StubVAE(torch.bfloat16, "tensor")
    with pytest.raises(ValueError, match="nonnegative"):
        pack_batch(
            batch(),
            vae,
            StubTokenizer(),
            SimpleNamespace(latent_patch_size=2, action_dim=8),
            policy=StubPolicy(),
            num_frames=9,
            fps=15.0,
            device="cpu",
            vae_encode_batch=-1,
        )
    assert vae.calls == []
