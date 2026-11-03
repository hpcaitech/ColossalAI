import math

import pytest
import torch
from safetensors.torch import save_file

from tests.kit.numerics import (
    Routers,
    Trace,
    assert_aligned,
    diagnose_routing,
    error_multiplier,
    flatten_tensors,
    judge,
    judge_equal,
    judge_relative_bound,
    require_gradients,
    run_fwd_bwd,
)

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


def test_trace_is_a_snapshot(tmp_path):
    # CPU tensors are copied: an in-place update after capture (e.g. an optimizer step) does not reach the trace
    weight = torch.tensor([1.0])
    storage = torch.arange(4.0)
    trace = Trace.capture({"w": weight, "a": storage, "view": storage[1:3]})
    weight.add_(1.0)
    storage.zero_()
    trace.save(tmp_path / "t.safetensors")
    for t in (trace, Trace.load(tmp_path / "t.safetensors")):
        assert t["w"].tolist() == [1.0]
        assert t["a"].tolist() == [0.0, 1.0, 2.0, 3.0] and t["view"].tolist() == [1.0, 2.0]


def test_comparing_nothing_fails():
    for report in (judge({}, {}, {}), judge_equal({}, {}), judge_relative_bound({}, {}, torch.bfloat16)):
        assert [(e.name, e.reason) for e in report.failures] == [("<all>", "nothing was compared")]


class _PartlyUsed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = torch.nn.Linear(4, 4)
        self.unused = torch.nn.Linear(4, 4)
        self.frozen = torch.nn.Linear(4, 4).requires_grad_(False)

    def forward(self, x):
        return self.frozen(self.used(x))


def test_missing_gradients_fail_unless_declared():
    inputs = {"x": torch.randn(2, 4, dtype=torch.float64)}
    runs = [run_fwd_bwd(_PartlyUsed(), inputs, dtype) for dtype in (torch.float32, torch.float32, torch.float64)]
    # no run records the unused gradients, so judging the recorded tensors alone passes
    assert judge(*runs).passed
    assert runs[0].meta["missing_grads"] == ["unused.bias", "unused.weight"]
    report = require_gradients(judge(*runs), test=runs[0], ref=runs[1], truth=runs[2])
    assert {(e.name, e.reason) for e in report.failures} == {
        (f"grad/param/unused.{p}", "no gradient in test, ref, truth") for p in ("weight", "bias")
    }
    # declared unused, and frozen parameters are never expected to have a gradient
    declared = run_fwd_bwd(_PartlyUsed(), inputs, torch.float32, unused_params=("unused.weight", "unused.bias"))
    assert declared.meta["missing_grads"] == []


class _TinyMoE(torch.nn.Module):
    """Top-2 of 4 experts, weighted by the softmax of the router scores."""

    def __init__(self, gate: type = torch.nn.Linear):
        super().__init__()
        self.gate = gate(8, 4, bias=False)
        self.experts = torch.nn.ModuleList(torch.nn.Linear(8, 8) for _ in range(4))

    def forward(self, x):
        weights, chosen = self.gate(x).softmax(-1).topk(2, dim=-1)
        out = torch.zeros_like(x)
        for k in range(2):
            for e, expert in enumerate(self.experts):
                out = out + (chosen[..., k : k + 1] == e) * weights[..., k : k + 1] * expert(x)
        return out


class _InvertedGate(torch.nn.Linear):
    """A routing bug: the first token of every sequence gets its scores negated, so it goes to the other two experts."""

    def forward(self, x):
        scores = super().forward(x)
        return torch.cat([-scores[:, :1], scores[:, 1:]], dim=1)


ROUTERS = Routers(select=lambda name, module: name == "gate", top_k=2)


def _moe_runs(test_module, routers=ROUTERS):
    torch.manual_seed(0)
    ref_module = _TinyMoE()
    test_module.load_state_dict(ref_module.state_dict())
    inputs = {"x": torch.randn(4, 16, 8, dtype=torch.float64)}
    truth = run_fwd_bwd(ref_module, inputs, torch.float64, routers=routers)
    ref = run_fwd_bwd(ref_module, inputs, torch.bfloat16, routers=routers)
    test = run_fwd_bwd(test_module, inputs, torch.bfloat16, routers=routers)
    return test, ref, truth


def test_routing_is_recorded_and_flips_counted():
    test, ref, truth = _moe_runs(_TinyMoE())
    assert [call["module"] for call in truth.meta["routing"]] == ["gate"]
    assert len(truth.meta["routing"][0]["experts"]) == 4 * 16
    report = diagnose_routing(judge(test, ref, truth), test, ref, truth)
    report.assert_pass()
    # the same code at the same precision: whatever the precision flips, it flips in both
    assert report.routing["routings"] == 64
    assert report.routing["test"] == report.routing["ref"]
    assert "routing flips against truth" in report.format()


def test_routing_flips_are_noted_on_failures_without_changing_verdicts():
    test, ref, truth = _moe_runs(_TinyMoE(gate=_InvertedGate))
    judged = judge(test, ref, truth)
    verdicts = {e.name: e.passed for e in judged.entries}
    report = diagnose_routing(judged, test, ref, truth)
    assert {e.name: e.passed for e in report.entries} == verdicts
    # the first token of each of the 4 sequences flips in test, on top of what ref flips elsewhere
    assert report.routing["test"] >= 4 and report.routing["test"] > report.routing["ref"]
    assert report.failures and all("routing:" in e.reason for e in report.failures)


def test_routers_that_match_nothing_fail():
    nothing = Routers(select=lambda name, module: False, top_k=2)
    test, ref, truth = _moe_runs(_TinyMoE(), routers=nothing)
    report = diagnose_routing(judge(test, ref, truth), test, ref, truth)
    assert [e.name for e in report.failures] == ["<routing>"]
    assert "Routers.select" in report.failures[0].reason


def test_routing_survives_save_and_load(tmp_path):
    test, ref, truth = _moe_runs(_TinyMoE())
    for name, trace in (("test", test), ("ref", ref), ("truth", truth)):
        trace.save(tmp_path / f"{name}.safetensors")
    loaded = [Trace.load(tmp_path / f"{name}.safetensors") for name in ("test", "ref", "truth")]
    report = diagnose_routing(judge(*loaded), *loaded)
    assert report.routing == diagnose_routing(judge(test, ref, truth), test, ref, truth).routing
