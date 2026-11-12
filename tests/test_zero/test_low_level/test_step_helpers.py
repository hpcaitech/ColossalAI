"""CPU contract tests for the helpers the ZeRO-2 step uses to keep its launches low.

``_compute_grad_norm`` replaces the per-shard float64 cast and norm with one multi-tensor norm whose pieces are
summed in float64; the cast-buffer helpers reuse one fp32 grad buffer and one working-dtype write-back buffer per
master param instead of allocating two tensors per parameter per step; ``_multi_tensor_copy`` falls back to
per-tensor copies when the backend has no foreach copy. The tests bind the methods to stubs, so they need no
optimizer, accelerator or process group.
"""

import pytest
import torch

from colossalai.zero.low_level.low_level_optim import LowLevelZeroOptimizer


class _NormStub:
    """Only the attributes ``_compute_grad_norm`` reads."""

    _GRAD_NORM_CHUNK = 64

    def __init__(self, weights=None):
        self._weights = weights or {}

    def _grad_norm_weight(self, grad):
        return self._weights.get(id(grad), 1.0)

    def _grad_norm_process_groups(self):
        return []


class _CastStub:
    """Only the state the cast-buffer helpers read; the helpers are bound so one can call the other."""

    _reused_buffer = LowLevelZeroOptimizer._reused_buffer
    _master_grad_buffer = LowLevelZeroOptimizer._master_grad_buffer
    _working_cast_buffer = LowLevelZeroOptimizer._working_cast_buffer
    _multi_tensor_copy = LowLevelZeroOptimizer._multi_tensor_copy

    def __init__(self, dtype=torch.bfloat16):
        self._master_grad_buffers = {}
        self._working_cast_buffers = {}
        self._dtype = dtype


def _reference(grads, norm_type=2.0):
    return sum(float(grad.data.double().norm(norm_type) ** norm_type) for grad in grads) ** (1.0 / norm_type)


@pytest.mark.parametrize("num_grads", [0, 1, 7])
def test_norm_matches_the_float64_reference(num_grads):
    generator = torch.Generator().manual_seed(num_grads + 3)
    grads = [torch.randn(97, generator=generator) for _ in range(num_grads)]

    value = LowLevelZeroOptimizer._compute_grad_norm(_NormStub(), grads, 2)

    assert value == pytest.approx(_reference(grads), rel=1e-6)


def test_a_long_shard_is_chunked_and_stays_accurate():
    generator = torch.Generator().manual_seed(0)
    grads = [torch.randn(1000, generator=generator) * 3.0 for _ in range(5)]

    value = LowLevelZeroOptimizer._compute_grad_norm(_NormStub(), grads, 2)

    assert value == pytest.approx(_reference(grads), rel=1e-6)


def test_the_multi_tensor_path_norms_every_chunk_in_one_call(monkeypatch):
    calls = []
    real_foreach_norm = torch._foreach_norm

    def counting_foreach_norm(tensors, ord):
        calls.append(len(tensors))
        return real_foreach_norm(tensors, ord)

    monkeypatch.setattr(torch, "_foreach_norm", counting_foreach_norm)
    grads = [torch.randn(100) for _ in range(3)]

    LowLevelZeroOptimizer._compute_grad_norm(_NormStub(), grads, 2)

    # three shards of 100 elements, chunk 64 -> two pieces each, all normed by one multi-tensor call
    assert calls == [6]


def test_a_shard_within_the_chunk_size_is_passed_through(monkeypatch):
    received = []
    real_foreach_norm = torch._foreach_norm

    def counting_foreach_norm(tensors, ord):
        received.extend(tensors)
        return real_foreach_norm(tensors, ord)

    monkeypatch.setattr(torch, "_foreach_norm", counting_foreach_norm)
    grads = [torch.randn(64) for _ in range(2)]

    LowLevelZeroOptimizer._compute_grad_norm(_NormStub(), grads, 2)

    # The shard is exactly one chunk: no view or split object may be built for it, the grad's own storage goes to
    # the norm.
    assert len(received) == 2
    assert all(tensor.data_ptr() == grad.data_ptr() for tensor, grad in zip(received, grads))


def test_non_fp32_grads_keep_the_per_shard_path(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "_foreach_norm", lambda *args: calls.append(args) or [])
    generator = torch.Generator().manual_seed(2)
    grads = [torch.randn(10, generator=generator).to(torch.bfloat16) for _ in range(2)]

    value = LowLevelZeroOptimizer._compute_grad_norm(_NormStub(), grads, 2)

    assert calls == [], "a bf16 shard has no wide norm to take; it must stay on the float64 path"
    assert value == pytest.approx(_reference(grads), rel=1e-6)


def test_non_contiguous_gradients_stay_correct():
    base = torch.randn(6, 8)

    value = LowLevelZeroOptimizer._compute_grad_norm(_NormStub(), [base.t()], 2)

    assert value == pytest.approx(_reference([base.t()]), rel=1e-6)


def test_weights_apply_per_shard_not_per_chunk():
    generator = torch.Generator().manual_seed(1)
    grads = [torch.randn(50, generator=generator), torch.randn(1000, generator=generator)]
    stub = _NormStub(weights={id(grads[1]): 0.5})

    value = LowLevelZeroOptimizer._compute_grad_norm(stub, grads, 2)

    expected = (float(grads[0].double().norm(2) ** 2) + 0.5 * float(grads[1].double().norm(2) ** 2)) ** 0.5
    assert value == pytest.approx(expected, rel=1e-6)


def test_infinity_norm_is_the_max_absolute_value():
    grads = [torch.tensor([1.0, -3.0]), torch.tensor([2.0])]

    value = LowLevelZeroOptimizer._compute_grad_norm(_NormStub(), grads, float("inf"))

    assert value == pytest.approx(3.0)


def test_cast_buffers_are_reused_and_filled():
    stub = _CastStub()
    master = torch.randn(32, 4)
    grad = master.to(torch.bfloat16)

    first = stub._master_grad_buffer(master)
    second = stub._master_grad_buffer(master)
    assert first is second, "the fp32 grad buffer must be reused across steps"
    stub._multi_tensor_copy([first], [grad])
    assert torch.equal(first, grad.float())

    cast_back = stub._working_cast_buffer(master)
    stub._multi_tensor_copy([cast_back], [master])
    assert cast_back.dtype is torch.bfloat16
    assert torch.equal(cast_back.float(), master.to(torch.bfloat16).float())


def test_a_resized_param_gets_a_fresh_buffer():
    stub = _CastStub()
    master = torch.randn(16)

    stale = stub._master_grad_buffer(master)
    master.data = master.data[:8]
    fresh = stub._master_grad_buffer(master)

    assert fresh is not stale
    assert fresh.shape == master.shape


def test_multi_tensor_copy_falls_back_to_per_tensor(monkeypatch):
    def unsupported(*args, **kwargs):
        raise NotImplementedError("no foreach copy on this backend")

    monkeypatch.setattr(torch, "_foreach_copy_", unsupported)
    master = torch.randn(8)
    destination = torch.empty(8, dtype=torch.bfloat16)

    _CastStub()._multi_tensor_copy([destination], [master])

    assert torch.equal(destination.float(), master.to(torch.bfloat16).float())
