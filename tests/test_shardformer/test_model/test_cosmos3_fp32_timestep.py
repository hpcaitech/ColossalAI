"""The timestep embedder's arithmetic runs in fp32 while its parameters stay in the training dtype.

The reference implementation embeds timesteps in fp32 deliberately -- "Timesteps are computed in FP32 for
numerical stability" -- with bf16 parameters, measured at its own module's forward. The policy reproduces that
without autocast, so these checks are about what the module computes with, not about what it stores.

CPU, no checkpoint, no device: the model here is a two-layer stand-in with the one attribute name the policy
looks for.
"""

import torch
import torch.nn as nn

from colossalai.shardformer.modeling.cosmos3 import install_fp32_timestep_embedding


class _Embedder(nn.Module):
    """Records the dtype its arithmetic ran in, which is the thing under test."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, dim)
        self.seen = None

    def forward(self, sample):
        hidden = self.linear_1(sample)
        self.seen = hidden.dtype
        return self.linear_2(hidden)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.time_embedder = _Embedder()
        self.other = nn.Linear(8, 8)


def _model_in(dtype):
    model = _Model().to(dtype)
    return model


def test_wraps_exactly_the_time_embedder():
    model = _model_in(torch.bfloat16)
    assert install_fp32_timestep_embedding(model) == 1


def test_a_model_without_one_is_reported_as_zero():
    class _NoEmbedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.other = nn.Linear(8, 8)

    assert install_fp32_timestep_embedding(_NoEmbedder()) == 0


def test_installing_twice_wraps_once():
    model = _model_in(torch.bfloat16)
    assert install_fp32_timestep_embedding(model) == 1
    assert install_fp32_timestep_embedding(model) == 0, "a second pass must not stack another wrapper"


def test_the_arithmetic_runs_in_fp32_with_bf16_parameters():
    model = _model_in(torch.bfloat16)
    sample = torch.randn(2, 8, dtype=torch.bfloat16)

    model.time_embedder(sample)
    assert model.time_embedder.seen == torch.bfloat16, "the unwrapped module should follow its parameters"

    install_fp32_timestep_embedding(model)
    out = model.time_embedder(sample)
    assert model.time_embedder.seen == torch.float32, "the wrapped module must compute in fp32"
    assert out.dtype == torch.bfloat16, "and hand back the caller's dtype, so nothing downstream changes"
    assert {p.dtype for p in model.time_embedder.parameters()} == {torch.bfloat16}, (
        "the parameters must stay in the training dtype: this reproduces bf16 storage with fp32 arithmetic, "
        "not fp32 storage"
    )


def test_the_gradient_reaches_the_bf16_parameters():
    """The fp32 copies are made with .float(), so the gradient has to flow back through that."""
    model = _model_in(torch.bfloat16)
    install_fp32_timestep_embedding(model)
    out = model.time_embedder(torch.randn(2, 8, dtype=torch.bfloat16))
    out.float().sum().backward()
    for name, param in model.time_embedder.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert param.grad.dtype == param.dtype, f"{name} grad is {param.grad.dtype}, not {param.dtype}"


def test_fp32_arithmetic_matches_exact_arithmetic_on_the_same_operands():
    """What the change buys, stated so it can be false.

    The comparison holds the operands fixed. Both the parameters and the input are already rounded to bf16 by
    the time this module runs -- the model casts the sinusoidal projection to the parameter dtype before
    calling it -- and no arithmetic can undo that. What fp32 arithmetic changes is everything after: the
    matmul, the activation and the accumulation.

    So the reference is those same rounded operands evaluated in fp32, and the claim is exact: computing in
    fp32 and rounding the result once lands on the reference rounded once, element for element, while
    computing in bf16 does not. The relative numbers are reported too, because the single remaining rounding
    of the output is the floor under both -- it is why the ratio between them is small even though one is
    exactly right and the other is not.
    """
    source = _Model().to(torch.bfloat16)  # the rounded operands, once
    sample_bf16 = torch.randn(64, 8).to(torch.bfloat16)

    exact_model = _Model().to(torch.float32)
    exact_model.load_state_dict({k: v.float() for k, v in source.state_dict().items()})

    wrapped = _Model().to(torch.bfloat16)
    wrapped.load_state_dict(source.state_dict())
    install_fp32_timestep_embedding(wrapped)

    with torch.no_grad():
        exact = exact_model.time_embedder(sample_bf16.float())
        plain = source.time_embedder(sample_bf16)
        upcast = wrapped.time_embedder(sample_bf16)

    rounded_reference = exact.to(torch.bfloat16)
    assert torch.equal(
        upcast, rounded_reference
    ), "fp32 arithmetic plus one rounding must equal the fp32 reference rounded once"
    assert not torch.equal(plain, rounded_reference), (
        "bf16 arithmetic reaching the same result would mean this module's precision does not matter here, "
        "and the test would be proving nothing"
    )

    def rel(x):
        return (x.float() - exact).norm().item() / exact.norm().item()

    assert rel(upcast) < rel(plain), f"fp32 arithmetic {rel(upcast):.3e} against bf16 {rel(plain):.3e}"
