"""Self-tests of the benchmark framework that need no device: statistics, FLOPs, the peak table, the report and its
template, and one full run of a tiny model on the CPU."""

import json

import pytest
import transformers

from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.benchmark import BenchmarkSpec, decoder_flops_per_token
from tests.kit.benchmark import report as report_module
from tests.kit.benchmark.flops import counted_flops_per_token, flops_per_token, recompute_countable
from tests.kit.benchmark.measure import CONFIGS, Setup
from tests.kit.benchmark.peak import PEAKS, lookup
from tests.kit.benchmark.report import build_report, percentile, render_markdown, summarize
from tests.kit.benchmark.run import run_benchmark


def tiny_llama():
    config = transformers.LlamaConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attn_implementation="eager",
        use_cache=False,
    )
    return transformers.LlamaForCausalLM(config)


# module-level functions only: the spec is sent to spawned processes
TINY = BenchmarkSpec(models={"tiny": tiny_llama}, seq_len=16, micro_batch=2, flops_fn=decoder_flops_per_token)
# 1 GFLOP per token against a 100 TFLOPS nominal and 80 TFLOPS measured peak
FLOPS = {"model_flops_per_token": 1e9, "hardware_flops_per_token": 1.25e9, "difference": 0.0, "source": "formula"}
PEAK = {"device": "X", "dtype": "bf16", "nominal_tflops": 100.0, "measured_tflops": 80.0}
NO_PEAK = {"device": "cpu", "dtype": "fp32", "nominal_tflops": None, "measured_tflops": None}
# the counter also sees the rotary embedding's frequency product (head_dim / 2 FLOPs per token, forward only), which
# the formula leaves out with the other non-weight operations: 2e-5 of this tiny model, 1e-9 of a real one
ROTARY_SHARE = 1e-4


def _run(times, setup=None, finite=True):
    return {
        "setup": setup or {"entry": "e", "size": "s", "config": "ddp_fp32"},
        "world_size": 2,
        "parameters": 1000,
        "optimizer": "AdamW",
        "tokens_per_step": 64,
        "samples_per_step": 4,
        "step_time_s": times,
        "memory_bytes": {"host_rss_peak": 1 << 30},
        "loss": {"first": 1.0, "last": 0.9, "finite": finite},
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
    summary = summarize(_run([1.0, 1.0, 1.0, 9.0]), FLOPS, PEAK)  # one stalled step does not move the headline
    assert summary["step_time_s"]["median"] == 1.0
    assert summary["tokens_per_s"] == 64
    assert summary["tokens_per_s_per_device"] == 32
    assert summary["samples_per_s"] == 4


def test_report_flags_unstable_repeats_and_picks_the_slower_middle():
    stable = build_report([_run([1.0, 1.0]), _run([1.02, 1.02])], {}, {"commit": "abc"}, ["cmd"], FLOPS, PEAK)
    assert stable["flags"] == []
    assert stable["headline_repeat"] == 1  # the slower of two
    assert stable["repeat_spread"] == pytest.approx(0.02)

    unstable = build_report([_run([1.0]), _run([1.2])], {}, {}, [], FLOPS, PEAK)
    assert len(unstable["flags"]) == 1 and "not stable" in unstable["flags"][0]

    diverged = build_report([_run([1.0], finite=False)], {}, {}, [], FLOPS, PEAK)
    assert any("not finite" in flag for flag in diverged["flags"])


def test_report_refuses_runs_of_different_setups():
    other = {"entry": "e", "size": "s", "config": "zero2_bf16"}
    with pytest.raises(ValueError, match="same setup"):
        build_report([_run([1.0]), _run([1.0], setup=other)], {}, {}, [], FLOPS, PEAK)
    with pytest.raises(ValueError):
        build_report([], {}, {}, [], FLOPS, PEAK)


def test_setup_and_spec_validation():
    with pytest.raises(ValueError, match="unknown config"):
        Setup(entry="e", size="s", config="ddp_bf16", seq_len=8, micro_batch=1)
    with pytest.raises(ValueError, match="timed step"):
        Setup(entry="e", size="s", config="ddp_fp32", seq_len=8, micro_batch=1, timed_steps=0)
    with pytest.raises(ValueError, match="at least one size"):
        BenchmarkSpec(models={})
    assert BenchmarkSpec(models={"a": tiny_llama, "b": tiny_llama}).default_size == "a"


@rerun_if_address_is_in_use()
def test_cpu_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    setup = Setup(
        entry="tiny", size="tiny", config="ddp_fp32", seq_len=16, micro_batch=2, warmup_steps=1, timed_steps=3
    )
    result = run_benchmark(setup, nproc=2, repeats=2, spec=TINY, command=["python", "-m", "x"])

    assert result["schema_version"] == report_module.SCHEMA_VERSION
    assert result["world_size"] == 2
    assert len(result["repeats"]) == 2
    for run in result["repeats"]:
        assert len(run["step_time_s"]) == 3 and all(t > 0 for t in run["step_time_s"])
        assert run["tokens_per_step"] == 2 * 2 * 16
        assert run["loss"]["finite"]
    # no allocator statistics on the CPU, only the host
    assert set(result["memory_bytes"]) == {"host_rss_peak"}
    assert result["environment"]["accelerator"] == "cpu"
    assert "commit" in result["repository"] or "error" in result["repository"]

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
    assert "Peak reserved | n/a" in markdown
    # no peak for a CPU: MFU is missing and the report says why, it is never left out silently
    assert result["throughput"]["mfu_nominal"] is None
    assert "MFU: no peak is known for cpu in fp32" in markdown
    assert abs(result["flops"]["difference"]) < ROTARY_SHARE


def test_mfu_divides_model_flops_by_both_peaks():
    summary = summarize(_run([1.0]), FLOPS, PEAK)  # 32 tokens/s per device × 1 GFLOP = 0.032 TFLOPS
    assert summary["model_tflops_per_device"] == pytest.approx(0.032)
    assert summary["mfu_nominal"] == pytest.approx(0.032 / 100)
    assert summary["mfu_measured"] == pytest.approx(0.032 / 80)
    assert summary["hfu_nominal"] == pytest.approx(0.04 / 100)
    assert summarize(_run([1.0]), FLOPS, NO_PEAK)["mfu_nominal"] is None


def test_report_flags_a_formula_that_disagrees_with_the_counted_flops():
    off = dict(FLOPS, difference=0.08)
    assert any("FLOPs formula" in flag for flag in build_report([_run([1.0])], {}, {}, [], off, PEAK)["flags"])
    assert build_report([_run([1.0])], {}, {}, [], dict(FLOPS, difference=None), PEAK)["flags"] == []


def test_formula_matches_the_counted_flops():
    formula = decoder_flops_per_token(tiny_llama(), 16)
    counted = counted_flops_per_token(TINY, "tiny", seq_len=16, micro_batch=2)
    assert counted == pytest.approx(formula, rel=ROTARY_SHARE)
    info = flops_per_token(TINY, "tiny", 16, 2, gradient_checkpointing=False)
    assert info["source"] == "formula" and info["hardware_flops_per_token"] == formula


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


def test_attention_flops_grow_with_the_sequence_and_are_not_halved():
    model = tiny_llama()
    # 12 · layers · heads · head_dim per extra token of context
    assert decoder_flops_per_token(model, 17) - decoder_flops_per_token(model, 16) == 12 * 2 * 4 * 8


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
