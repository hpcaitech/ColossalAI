"""Self-tests of the benchmark framework that need no device: the config, the metrics from hand-made raw records,
FLOPs, the peak table, the report, and one full run of a tiny model on the CPU."""

import json
import math
from dataclasses import asdict

import pytest
import torch
import transformers

from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.benchmark import BenchmarkSpec, decoder_flops_per_token
from tests.kit.benchmark.config.model import DEFAULT_IMPLEMENTATION
from tests.kit.benchmark.config.plugins import CONFIGS
from tests.kit.benchmark.config.setup import Setup, parse_field
from tests.kit.benchmark.metrics.compute import (
    PEAKS,
    AttentionCounter,
    counted_flops_per_token,
    flops_per_token,
    lookup,
    recompute_countable,
    utilization,
)
from tests.kit.benchmark.metrics.memory import GiB
from tests.kit.benchmark.metrics.summary import summarize
from tests.kit.benchmark.metrics.time import percentile, slowest_rank_steps, time_metrics
from tests.kit.benchmark.runner.measure import COMPLETED
from tests.kit.benchmark.scenarios import report as report_module
from tests.kit.benchmark.scenarios.report import build_report, render_markdown
from tests.kit.benchmark.scenarios.single import run_single


def tiny_llama(attn_implementation="sdpa"):
    config = transformers.LlamaConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attn_implementation=attn_implementation,
        use_cache=False,
    )
    return transformers.LlamaForCausalLM(config)


def unchanged(model):
    """An implementation that computes exactly what the default one does."""


def not_on_meta(model):
    """An implementation that, like a fused kernel, cannot run on the meta device."""
    if next(model.parameters()).is_meta:
        raise NotImplementedError("this kernel has no meta implementation")


# module-level functions only: the spec is sent to spawned processes
TINY = BenchmarkSpec(
    models={"tiny": tiny_llama},
    seq_len=16,
    micro_batch=2,
    causal_attention=True,
    flops_fn=decoder_flops_per_token,
    implementations={"unchanged": unchanged, "fused": not_on_meta},
)
SETUP = asdict(Setup(entry="e", size="s", config="ddp_fp32", seq_len=16, micro_batch=2))
# 1 GFLOP per token against a 100 TFLOPS nominal and 80 TFLOPS measured peak
FLOPS = {"model_flops_per_token": 1e9, "hardware_flops_per_token": 1.25e9, "difference": 0.0, "source": "formula"}
PEAK = {"device": "X", "dtype": "bf16", "nominal_tflops": 100.0, "measured_tflops": 80.0}
NO_PEAK = {"device": "cpu", "dtype": "fp32", "nominal_tflops": None, "measured_tflops": None}
# the counter also sees the rotary embedding's frequency product (head_dim / 2 FLOPs per token, forward only), which
# the formula leaves out with the other non-weight operations: 2e-5 of this tiny model, 1e-9 of a real one
ROTARY_SHARE = 1e-4


def record(times, forward_backward=None, setup=None, finite=True, memory=None, used=None, ranks=2):
    """A completed ``runner.measure`` record of ``ranks`` identical ranks, each training 32 tokens (2 samples) per
    step in ``times`` seconds, of which ``forward_backward`` until the end of the backward (default 80%)."""
    rank = {
        "tokens": 32,
        "samples": 2,
        "step_s": list(times),
        "forward_backward_s": list(forward_backward or [t * 0.8 for t in times]),
        "losses": [1.0, 0.9 if finite else math.nan],
        "memory": memory or {"host_rss_peak": 1 << 30},
        "timeline": None,
        "device_used_at_start": used,
    }
    return {
        "status": COMPLETED,
        "setup": setup or SETUP,
        "parameters": 1000,
        "optimizer": "AdamW",
        "environment": {"accelerator": "cpu"},
        "ranks": [dict(rank) for _ in range(ranks)],
    }


def test_percentile_interpolates_like_numpy():
    values = [4.0, 1.0, 3.0, 2.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 50) == 2.5
    assert percentile(values, 10) == pytest.approx(1.3)
    assert percentile([7.0], 90) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_throughput_comes_from_the_median_step():
    time = time_metrics(record([1.0, 1.0, 1.0, 9.0]))  # one stalled step does not move the headline
    assert time["step_time_s"]["median"] == 1.0
    assert time["tokens_per_step"] == 64
    assert time["tokens_per_s"] == 64
    assert time["tokens_per_s_per_device"] == 32
    assert time["samples_per_s"] == 4


def test_each_step_is_its_slowest_rank_and_splits_there():
    run = record([1.0, 1.0])
    run["ranks"][1] = dict(run["ranks"][1], step_s=[2.0, 0.5], forward_backward_s=[1.5, 0.1])
    steps, forward_backward = slowest_rank_steps(run)
    assert steps == [2.0, 1.0] and forward_backward == [1.5, 0.8]
    split = time_metrics(record([1.0, 1.0, 1.0], forward_backward=[0.75, 0.75, 0.75]))
    assert split["forward_backward_share"] == 0.75 and split["optimizer_share"] == 0.25


def test_summary_flags_unstable_repeats_and_picks_the_slower_middle():
    stable = summarize([record([1.0, 1.0]), record([1.02, 1.02])], FLOPS, PEAK)
    assert stable["flags"] == []
    assert stable["headline_repeat"] == 1  # the slower of two
    assert stable["repeat_spread"] == pytest.approx(0.02)

    unstable = summarize([record([1.0]), record([1.2])], FLOPS, PEAK)
    assert len(unstable["flags"]) == 1 and "not stable" in unstable["flags"][0]

    diverged = summarize([record([1.0], finite=False)], FLOPS, PEAK)
    assert any("not finite" in flag for flag in diverged["flags"])


def test_summary_refuses_runs_of_different_setups():
    other = dict(SETUP, config="zero2_bf16")
    with pytest.raises(ValueError, match="same setup"):
        summarize([record([1.0]), record([1.0], setup=other)], FLOPS, PEAK)
    with pytest.raises(ValueError):
        summarize([], FLOPS, PEAK)


def test_setup_and_spec_validation():
    with pytest.raises(ValueError, match="unknown config"):
        Setup(entry="e", size="s", config="ddp_bf16", seq_len=8, micro_batch=1)
    with pytest.raises(ValueError, match="timed step"):
        Setup(entry="e", size="s", config="ddp_fp32", seq_len=8, micro_batch=1, timed_steps=0)
    with pytest.raises(ValueError, match="one device"):
        Setup(entry="e", size="s", config="ddp_fp32", seq_len=8, micro_batch=1, devices=0)
    with pytest.raises(ValueError, match="at least one size"):
        BenchmarkSpec(models={})
    with pytest.raises(ValueError, match="name the other"):
        BenchmarkSpec(models={"a": tiny_llama}, implementations={DEFAULT_IMPLEMENTATION: unchanged})
    with pytest.raises(ValueError, match="reference implementation"):
        BenchmarkSpec(models={"a": tiny_llama}, reference_implementation="fused")
    assert BenchmarkSpec(models={"a": tiny_llama, "b": tiny_llama}).default_size == "a"
    assert TINY.implementation_names == (DEFAULT_IMPLEMENTATION, "unchanged", "fused")
    with pytest.raises(ValueError, match="no implementation"):
        TINY.build("tiny", "triton")
    assert parse_field("config", "pytorch_fsdp2_bf16") == "pytorch_fsdp2_bf16"
    assert parse_field("seq_len", "8192") == 8192 and parse_field("gradient_checkpointing", "True") is True
    with pytest.raises(ValueError):
        parse_field("hidden_size", "1")


def test_build_applies_the_implementation_then_prepare():
    calls = []

    def implementation(model):
        calls.append("implementation")
        return torch.nn.Linear(1, 1)

    spec = BenchmarkSpec(
        models={"tiny": tiny_llama},
        implementations={"other": implementation},
        prepare_fn=lambda model: calls.append(type(model).__name__),
    )
    assert isinstance(spec.build("tiny", "other"), torch.nn.Linear)
    assert calls == ["implementation", "Linear"]
    assert isinstance(spec.build("tiny"), transformers.LlamaForCausalLM)


@rerun_if_address_is_in_use()
def test_cpu_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    setup = Setup(
        entry="tiny",
        size="tiny",
        config="ddp_fp32",
        seq_len=16,
        micro_batch=2,
        devices=2,
        implementation="unchanged",
        warmup_steps=1,
        timed_steps=3,
    )
    measurement = run_single(setup, repeats=2, spec=TINY)
    result = build_report(measurement, {"commit": "abc"}, ["python", "-m", "x"])

    assert result["schema_version"] == report_module.SCHEMA_VERSION and result["scenario"] == report_module.SINGLE
    summary = measurement["summary"]
    assert summary["devices"] == 2 and len(measurement["runs"]) == 2
    for run in measurement["runs"]:
        assert run["status"] == COMPLETED and len(run["ranks"]) == 2
        for rank in run["ranks"]:
            assert len(rank["step_s"]) == 3 and all(
                0 < f < s for f, s in zip(rank["forward_backward_s"], rank["step_s"])
            )
            assert rank["tokens"] == 2 * 16 and len(rank["losses"]) == 4
    assert summary["time"]["tokens_per_step"] == 2 * 2 * 16
    assert 0 < summary["time"]["forward_backward_share"] < 1
    # no allocator statistics on the CPU, only the host
    assert set(summary["memory_bytes"]) == {"host_rss_peak"}
    assert result["environment"]["accelerator"] == "cpu"
    # counted in the reference implementation, whatever implementation is measured
    assert measurement["flops"]["counted_implementation"] == DEFAULT_IMPLEMENTATION

    path = tmp_path / "r.json"
    path.write_text(json.dumps(result))
    markdown = render_markdown(json.loads(path.read_text()))
    for heading in (
        "## Summary",
        "## Configuration",
        "## Metrics",
        "## Environment and reproduction",
        "## Not covered",
    ):
        assert heading in markdown
    assert "(tiny, unchanged implementation)" in markdown and "| Step split | forward + backward" in markdown
    assert "Peak reserved | n/a" in markdown
    # no peak for a CPU: MFU is missing and the report says why, it is never left out silently
    assert summary["utilization"]["mfu_nominal"] is None
    assert "MFU: no peak is known for cpu in fp32" in markdown
    assert abs(measurement["flops"]["difference"]) < ROTARY_SHARE


def test_mfu_divides_model_flops_by_both_peaks():
    result = utilization(32, FLOPS, PEAK)  # 32 tokens/s per device × 1 GFLOP = 0.032 TFLOPS
    assert result["model_tflops_per_device"] == pytest.approx(0.032)
    assert result["mfu_nominal"] == pytest.approx(0.032 / 100)
    assert result["mfu_measured"] == pytest.approx(0.032 / 80)
    assert result["hfu_nominal"] == pytest.approx(0.04 / 100)
    assert utilization(32, FLOPS, NO_PEAK)["mfu_nominal"] is None
    assert summarize([record([1.0])], FLOPS, PEAK)["utilization"] == result


def test_summary_flags_mfu_above_the_peak():
    fast = record([1e-5])  # 3.2e6 tokens/s per device × 1 GFLOP = 3.2 PFLOPS against 100 TFLOPS
    assert any("above the device's peak" in flag for flag in summarize([fast], FLOPS, PEAK)["flags"])


def test_summary_flags_a_formula_that_disagrees_with_the_counted_flops():
    off = dict(FLOPS, difference=0.08)
    assert any("FLOPs formula" in flag for flag in summarize([record([1.0])], off, PEAK)["flags"])
    assert summarize([record([1.0])], dict(FLOPS, difference=None), PEAK)["flags"] == []


def test_formula_matches_the_counted_flops():
    formula = decoder_flops_per_token(tiny_llama(), 16)
    counted = counted_flops_per_token(TINY, "tiny", seq_len=16, micro_batch=2)
    assert counted == pytest.approx(formula, rel=ROTARY_SHARE)
    info = flops_per_token(TINY, "tiny", 16, 2, gradient_checkpointing=False)
    assert info["source"] == "formula" and info["hardware_flops_per_token"] == formula


def test_flops_are_counted_in_the_reference_implementation():
    # the default reference can be counted although "fused" cannot run on the meta device
    assert flops_per_token(TINY, "tiny", 16, 2, gradient_checkpointing=False)["counted"] > 0
    fused_reference = BenchmarkSpec(
        models={"tiny": tiny_llama}, implementations={"fused": not_on_meta}, reference_implementation="fused"
    )
    with pytest.raises(NotImplementedError, match="meta"):
        counted_flops_per_token(fused_reference, "tiny", seq_len=16, micro_batch=2)


@pytest.mark.skipif(not recompute_countable(), reason="this torch cannot recompute on the meta device")
def test_recomputation_adds_a_forward():
    counted = counted_flops_per_token(TINY, "tiny", seq_len=16, micro_batch=2)
    recomputed = counted_flops_per_token(TINY, "tiny", seq_len=16, micro_batch=2, recompute=True)
    # the decoder layers' forward is done twice, the output projection once: between 1 and 4/3 of the model FLOPs
    assert counted < recomputed < counted * 4 / 3
    assert flops_per_token(TINY, "tiny", 16, 2, gradient_checkpointing=True)["hardware_flops_per_token"] == recomputed


@pytest.mark.skipif(recompute_countable(), reason="this torch can recompute on the meta device")
def test_recomputation_that_cannot_be_counted_says_so():
    with pytest.raises(NotImplementedError, match="hardware FLOPs"):
        counted_flops_per_token(TINY, "tiny", seq_len=16, micro_batch=2, recompute=True)


def test_causal_attention_flops_are_halved():
    model = tiny_llama()
    # 12 · layers · heads · head_dim per extra token of context, halved for the causal mask
    assert decoder_flops_per_token(model, 18) - decoder_flops_per_token(model, 16) == 2 * 12 * 2 * 4 * 8 // 2
    # one call over (batch 2, 4 heads, 16 queries, 16 keys, head_dim 8): 2 · 2·4·16·16 · (8 + 8) forward
    full, causal = AttentionCounter(causal=False), AttentionCounter(causal=True)
    for counter in (full, causal):
        q = torch.empty(2, 4, 16, 8, requires_grad=True)
        counter(q, torch.empty(2, 4, 16, 8), torch.empty(2, 4, 16, 8)).sum().backward()
    assert full.flops == 3 * 2 * 2 * 4 * 16 * 16 * 16 and causal.flops == full.flops // 2
    flagged = AttentionCounter(causal=False)
    flagged(torch.empty(1, 1, 4, 2), torch.empty(1, 1, 4, 2), torch.empty(1, 1, 4, 2), is_causal=True)
    assert flagged.flops == 2 * 4 * 4 * 4 // 2


def test_peak_lookup():
    assert lookup("NVIDIA H20", "bf16").nominal_tflops == 148.0
    assert lookup("NVIDIA H20-3e", "bf16").nominal_tflops == 148.0
    assert lookup("Ascend910B2", "fp16").nominal_tflops == 376.0
    # other 910B variants have less compute; an unknown device gives no peak rather than a guessed one
    assert lookup("Ascend910B3", "fp16") is None
    assert lookup("NVIDIA H100 80GB HBM3", "bf16") is None
    for peak in PEAKS:
        assert peak.nominal_source and peak.measured_tflops <= peak.nominal_tflops
    assert {config.dtype for config in CONFIGS.values()} <= {peak.dtype for peak in PEAKS}


def test_summary_flags_a_device_that_another_process_uses():
    memory = {"device_total": 100 * GiB}
    busy = summarize([record([1.0], memory=memory, used=int(35.8 * GiB))], FLOPS, PEAK)
    assert any("another process shared it" in flag for flag in busy["flags"])
    assert summarize([record([1.0], memory=memory, used=int(0.4 * GiB))], FLOPS, PEAK)["flags"] == []
