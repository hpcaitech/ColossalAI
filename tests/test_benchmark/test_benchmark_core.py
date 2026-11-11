"""Self-tests of the benchmark framework that need no device: statistics, the report and its template, and one full
run of a tiny model on the CPU."""

import json

import pytest
import transformers

from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.benchmark import BenchmarkSpec
from tests.kit.benchmark import report as report_module
from tests.kit.benchmark.measure import Setup
from tests.kit.benchmark.report import build_report, percentile, render_markdown, summarize
from tests.kit.benchmark.run import run_benchmark


def tiny_llama():
    config = transformers.LlamaConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        attn_implementation="eager",
        use_cache=False,
    )
    return transformers.LlamaForCausalLM(config)


# module-level functions only: the spec is sent to spawned processes
TINY = BenchmarkSpec(models={"tiny": tiny_llama}, seq_len=16, micro_batch=2)


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
    summary = summarize(_run([1.0, 1.0, 1.0, 9.0]))  # one stalled step does not move the headline
    assert summary["step_time_s"]["median"] == 1.0
    assert summary["tokens_per_s"] == 64
    assert summary["tokens_per_s_per_device"] == 32
    assert summary["samples_per_s"] == 4


def test_report_flags_unstable_repeats_and_picks_the_slower_middle():
    stable = build_report([_run([1.0, 1.0]), _run([1.02, 1.02])], {}, {"commit": "abc"}, ["cmd"])
    assert stable["flags"] == []
    assert stable["headline_repeat"] == 1  # the slower of two
    assert stable["repeat_spread"] == pytest.approx(0.02)

    unstable = build_report([_run([1.0]), _run([1.2])], {}, {}, [])
    assert len(unstable["flags"]) == 1 and "not stable" in unstable["flags"][0]

    diverged = build_report([_run([1.0], finite=False)], {}, {}, [])
    assert any("not finite" in flag for flag in diverged["flags"])


def test_report_refuses_runs_of_different_setups():
    other = {"entry": "e", "size": "s", "config": "zero2_bf16"}
    with pytest.raises(ValueError, match="same setup"):
        build_report([_run([1.0]), _run([1.0], setup=other)], {}, {}, [])
    with pytest.raises(ValueError):
        build_report([], {}, {}, [])


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
    assert "MFU" in markdown  # listed as not covered, never left out silently
