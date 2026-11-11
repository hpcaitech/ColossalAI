"""The sinusoidal features reach the timestep embedder unrounded.

The model's forward is ``time_embedder(time_proj(timesteps).to(time_embedder_dtype))``: the projection runs
in fp32 and is then rounded to the embedder's parameter dtype, which a mixed-precision plugin makes bf16.
The reference implementation rounds nothing there -- one module builds the features and feeds its own MLP,
under an fp32 autocast region, and only the output is cast. These checks are about which values the embedder
is handed, so the stand-in below reproduces that call shape and records them.

CPU, no checkpoint, no device.
"""

import pytest
import torch
import torch.nn as nn

from colossalai.shardformer.modeling.cosmos3 import (
    install_fp32_sinusoidal_timestep_input,
    install_fp32_timestep_embedding,
)


class _Proj(nn.Module):
    """Sinusoidal features, in fp32 whatever the module's dtype, as diffusers' Timesteps does."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.exp(-torch.arange(dim // 2, dtype=torch.float32)))

    def forward(self, timesteps):
        args = timesteps.float()[:, None] * self.frequencies.float()[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _Embedder(nn.Module):
    """Records the values it was handed, which is the thing under test."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim)
        self.seen = None

    def forward(self, sample):
        self.seen = sample
        return self.linear_1(sample)


class _Model(nn.Module):
    """The two submodules and the call that rounds between them."""

    def __init__(self) -> None:
        super().__init__()
        self.time_proj = _Proj()
        self.time_embedder = _Embedder()

    def forward(self, timesteps):
        dtype = next(self.time_embedder.parameters()).dtype
        return self.time_embedder(self.time_proj(timesteps).to(dtype))


def _timesteps():
    # Fractional values on purpose: whole numbers of this size survive bf16 and would hide the rounding.
    return torch.tensor([500.0, 123.456, 1.0 / 3.0], dtype=torch.float32)


def _installed(dtype=torch.bfloat16):
    """A model with both halves installed, which is the only supported order."""
    model = _Model().to(dtype)
    install_fp32_timestep_embedding(model)
    install_fp32_sinusoidal_timestep_input(model)
    return model


def test_wraps_the_pair():
    model = _Model().to(torch.bfloat16)
    install_fp32_timestep_embedding(model)
    assert install_fp32_sinusoidal_timestep_input(model) == 1


def test_it_refuses_to_install_without_the_fp32_arithmetic_wrapper():
    # Alone it would hand fp32 features to bf16 parameters, which fails in the first matmul; better to say
    # so at install time than at the first forward.
    with pytest.raises(RuntimeError, match="install_fp32_timestep_embedding"):
        install_fp32_sinusoidal_timestep_input(_Model().to(torch.bfloat16))


def test_a_model_without_the_pair_is_reported_as_zero():
    class _Half(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_embedder = _Embedder()

    assert install_fp32_sinusoidal_timestep_input(_Half()) == 0


def test_installing_twice_wraps_once():
    model = _installed()
    assert install_fp32_sinusoidal_timestep_input(model) == 0


def test_without_the_install_the_embedder_sees_the_rounded_copy():
    model = _Model().to(torch.bfloat16)
    model(_timesteps())
    assert model.time_embedder.seen.dtype == torch.bfloat16


def test_the_embedder_is_handed_the_projection_as_computed():
    model = _installed()
    expected = model.time_proj(_timesteps())
    model(_timesteps())
    seen = model.time_embedder.seen
    assert seen.dtype == torch.float32
    assert torch.equal(seen, expected)
    # And the rounding it replaces was a real one on this input, so the check above is not vacuous.
    assert not torch.equal(expected, expected.to(torch.bfloat16).float())


def test_it_composes_with_the_fp32_arithmetic_wrapper():
    model = _installed()
    out = model(_timesteps())
    seen = model.time_embedder.seen
    # fp32 features into fp32 arithmetic, and the result still leaves in the model's dtype.
    assert seen.dtype == torch.float32
    assert out.dtype == torch.bfloat16


def test_it_raises_rather_than_falling_back_when_the_pair_is_broken():
    model = _installed()
    with pytest.raises(RuntimeError, match="not handed the sinusoidal projection"):
        # A caller that embeds something else entirely: silently rounding it would be invisible.
        model.time_proj(_timesteps())
        model.time_embedder(torch.zeros(3, 8, dtype=torch.bfloat16))


def test_an_fp32_model_is_unchanged_by_it():
    model = _installed(torch.float32)  # the cast in the model's forward is already a no-op here
    expected = model.time_proj(_timesteps())
    model(_timesteps())
    assert torch.equal(model.time_embedder.seen, expected)
