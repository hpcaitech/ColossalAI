import math

import pytest
import torch
from safetensors.torch import save_file

from tests.kit.numerics import Trace, assert_aligned, error_multiplier, flatten_tensors, judge, run_fwd_bwd

HIDDEN = 256


def linear(x, w):
    return x @ w.t()


def linear_split_k(x, w):
    # same math, different summation order: what a fused or sharded kernel legitimately does
    half = HIDDEN // 2
    return x[:, :half] @ w[:, :half].t() + x[:, half:] @ w[:, half:].t()


def linear_off_by_one(x, w):
    # a typical sharding bug: one column of the reduction dimension is dropped
    half = HIDDEN // 2
    return x[:, :half] @ w[:, :half].t() + x[:, half + 1 :] @ w[:, half + 1 :].t()


def _inputs():
    g = torch.Generator().manual_seed(0)
    return {
        "x": torch.randn(64, HIDDEN, generator=g, dtype=torch.float64),
        "w": torch.randn(128, HIDDEN, generator=g, dtype=torch.float64),
    }


def _three_runs(test_fn, dtype=torch.bfloat16):
    inputs = _inputs()
    truth = run_fwd_bwd(linear, inputs, torch.float64)
    ref = run_fwd_bwd(linear, inputs, dtype)
    test = run_fwd_bwd(test_fn, inputs, dtype)
    return test, ref, truth


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_reordered_summation_is_within_baseline(dtype):
    test, ref, truth = _three_runs(linear_split_k, dtype)
    report = assert_aligned(test, ref, truth)
    assert {e.name for e in report.entries} == {"out", "grad/x", "grad/w"}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dropped_column_is_caught(dtype):
    test, ref, truth = _three_runs(linear_off_by_one, dtype)
    report = judge(test, ref, truth)
    failed = {e.name for e in report.failures}
    assert {"out", "grad/x", "grad/w"} <= failed
    assert all(e.ratio > 3 for e in report.failures)
    with pytest.raises(AssertionError, match="grad/w"):
        report.assert_pass()


def test_baseline_must_come_from_the_same_precision_recipe():
    # a float32 "reference" has a much smaller baseline error than bf16, so a correct bf16 kernel looks broken against it
    inputs = _inputs()
    truth = run_fwd_bwd(linear, inputs, torch.float64)
    mismatched_ref = run_fwd_bwd(linear, inputs, torch.float32)
    test = run_fwd_bwd(linear_split_k, inputs, torch.bfloat16)
    assert not judge(test, mismatched_ref, truth).passed


def test_module_params_get_gradients_and_same_weights():
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 16))
    inputs = {"input": torch.randn(8, 32, dtype=torch.float64)}
    truth = run_fwd_bwd(mlp, inputs, torch.float64)
    ref = run_fwd_bwd(mlp, inputs, torch.bfloat16)
    assert "grad/param/0.weight" in truth and "grad/input" in truth
    assert_aligned(ref, ref, truth)
    assert next(mlp.parameters()).dtype == torch.float32  # the caller's module is not modified


def test_error_multiplier_follows_pytorch():
    assert error_multiplier(torch.bfloat16, 10**6) == 3.0
    assert error_multiplier(torch.float16, 10**6) == 3.0
    assert error_multiplier(torch.float32, 10**6) == 2.0
    assert error_multiplier(torch.float32, 100) == 3.0
    assert error_multiplier(torch.float32, 100, small_tensor_multiplier=True) == 8.0
    assert error_multiplier(torch.bfloat16, 4, small_tensor_multiplier=True) == 10.0


def test_structural_mismatches_fail_with_reason():
    t = torch.ones(4)
    report = judge({"a": t, "extra": t}, {"a": t}, {"a": t, "b": t})
    reasons = {e.name: e.reason for e in report.failures}
    assert reasons == {"b": "missing in test", "extra": "missing in truth"}

    report = judge({"a": torch.ones(2, 2)}, {"a": torch.ones(4)}, {"a": torch.ones(4)})
    assert report.failures[0].reason.startswith("shape")


def test_non_finite_and_integer_tensors():
    truth = {"y": torch.ones(8, dtype=torch.float64), "idx": torch.arange(8)}
    ref = {"y": torch.ones(8, dtype=torch.bfloat16), "idx": torch.arange(8)}
    nan_test = {"y": torch.full((8,), math.nan, dtype=torch.bfloat16), "idx": torch.arange(8)}
    report = judge(nan_test, ref, truth)
    assert [e.reason for e in report.failures] == ["non-finite test output"]

    wrong_idx = {"y": torch.ones(8, dtype=torch.bfloat16), "idx": torch.arange(8).flip(0)}
    assert [e.name for e in judge(wrong_idx, ref, truth).failures] == ["idx"]


def test_flatten_tensors_names():
    nested = {"logits": torch.ones(1), "past": [torch.ones(1), None, (torch.ones(1),)]}
    assert sorted(flatten_tensors(nested)) == ["logits", "past/0", "past/2/0"]
    assert list(flatten_tensors(torch.ones(1))) == ["output"]


def test_trace_roundtrip_is_bit_exact(tmp_path):
    traces = {
        "truth": run_fwd_bwd(linear, _inputs(), torch.float64, role="truth"),
        "ref": run_fwd_bwd(linear, _inputs(), torch.bfloat16, role="ref"),
        "mixed": Trace.capture({"idx": torch.arange(4), "y": torch.ones(2, dtype=torch.float16)}, role="mixed"),
    }
    for role, trace in traces.items():
        path = tmp_path / f"{role}.safetensors"
        trace.save(path)
        loaded = Trace.load(path)
        assert loaded.meta == trace.meta
        assert set(loaded) == set(trace)
        for name in trace:
            assert loaded[name].dtype == trace[name].dtype
            assert torch.equal(loaded[name], trace[name])
    assert traces["ref"].meta["dtype"] == "torch.bfloat16"
    assert {"created", "torch", "transformers", "fn"} <= set(traces["ref"].meta)


def test_judging_saved_traces_matches_in_memory(tmp_path):
    runs = _three_runs(linear_off_by_one)
    loaded = []
    for i, trace in enumerate(runs):
        trace.save(tmp_path / f"{i}.safetensors")
        loaded.append(Trace.load(tmp_path / f"{i}.safetensors"))
    in_memory, offline = judge(*runs), judge(*loaded)
    assert [(e.name, e.passed, e.ratio) for e in in_memory.entries] == [
        (e.name, e.passed, e.ratio) for e in offline.entries
    ]


def test_load_rejects_plain_safetensors(tmp_path):
    save_file({"a": torch.ones(1)}, tmp_path / "plain.safetensors")
    with pytest.raises(ValueError, match="not a numerics trace"):
        Trace.load(tmp_path / "plain.safetensors")
