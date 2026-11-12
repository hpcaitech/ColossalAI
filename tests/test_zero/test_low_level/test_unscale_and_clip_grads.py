"""CPU contract tests for the multi-tensor path in ``LowLevelZeroOptimizer._unscale_and_clip_grads``.

The method only needs the arithmetic below, so the tests bind it to a stub instead of building a full optimizer or
a process group. They pin two properties: the scaling is the same as the original per-tensor loop, and the
``torch._foreach_mul_`` fast path is used when it is available but never allowed to change the result.
"""

import pytest
import torch

from colossalai.zero.low_level.low_level_optim import LowLevelZeroOptimizer


class _Stub:
    """Only the attributes ``_unscale_and_clip_grads`` reads."""

    def __init__(self, clip_grad_norm: float = 0.0):
        self.mixed_precision_mixin = None
        self._clip_grad_norm = clip_grad_norm


def _reference_scale(total_norm: float, clip_grad_norm: float) -> float:
    """The scale factor the original implementation computed inside the loop."""
    div_scale = 1.0
    if clip_grad_norm > 0.0:
        clip = (total_norm / div_scale + 1e-6) / clip_grad_norm
        if clip > 1:
            div_scale = clip * div_scale
    return 1.0 / div_scale


@pytest.mark.parametrize("clip_grad_norm", [0.0, 1.0])
@pytest.mark.parametrize("num_grads", [0, 1, 4])
def test_scaling_matches_the_per_tensor_loop(num_grads, clip_grad_norm):
    generator = torch.Generator().manual_seed(num_grads + 1)
    grads = [torch.randn(17, generator=generator) * 8.0 for _ in range(num_grads)]
    expected = [grad.clone() for grad in grads]

    LowLevelZeroOptimizer._unscale_and_clip_grads(_Stub(clip_grad_norm), grads, total_norm=3.0)

    scale = _reference_scale(3.0, clip_grad_norm)
    for grad, reference in zip(grads, expected):
        assert torch.equal(grad, reference * scale)


def test_the_multi_tensor_path_is_used_and_falls_back_cleanly(monkeypatch):
    calls = []
    real_foreach_mul_ = torch._foreach_mul_

    def counting_foreach_mul_(tensors, scalar):
        calls.append(len(tensors))
        return real_foreach_mul_(tensors, scalar)

    monkeypatch.setattr(torch, "_foreach_mul_", counting_foreach_mul_)
    grads = [torch.randn(5) for _ in range(3)]
    expected = [grad.clone() for grad in grads]
    LowLevelZeroOptimizer._unscale_and_clip_grads(_Stub(), grads, total_norm=0.0)

    assert calls == [3], "homogeneous grads must take one multi-tensor launch, not one launch per shard"
    for grad, reference in zip(grads, expected):
        assert torch.equal(grad, reference)

    # A backend without a foreach kernel must still scale every tensor exactly once.
    def unsupported(tensors, scalar):
        raise RuntimeError("not implemented for device")

    monkeypatch.setattr(torch, "_foreach_mul_", unsupported)
    grads = [torch.randn(5) for _ in range(2)]
    expected = [grad.clone() for grad in grads]
    LowLevelZeroOptimizer._unscale_and_clip_grads(_Stub(clip_grad_norm=1.0), grads, total_norm=2.0)

    scale = _reference_scale(2.0, 1.0)
    for grad, reference in zip(grads, expected):
        assert torch.equal(grad, reference * scale)


def test_mixed_dtype_gradients_stay_on_the_per_tensor_path(monkeypatch):
    fp32 = torch.randn(4)
    bf16 = torch.randn(4).to(torch.bfloat16)
    grads = [fp32, bf16]
    expected = [fp32.clone(), bf16.clone()]
    calls = []

    monkeypatch.setattr(torch, "_foreach_mul_", lambda *args: calls.append(args))
    LowLevelZeroOptimizer._unscale_and_clip_grads(_Stub(), grads, total_norm=0.0)

    assert calls == [], "a mixed dtype list cannot share one multi-tensor kernel"
    assert torch.equal(fp32, expected[0])
    assert torch.equal(bf16, expected[1])
