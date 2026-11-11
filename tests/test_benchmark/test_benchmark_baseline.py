"""Self-tests of the built-in baseline that need no device: which modules FSDP2 shards, and the comparison in the
report."""

import copy
from dataclasses import asdict

import pytest
from torch import nn

from tests.kit.benchmark.baseline import transformer_blocks
from tests.kit.benchmark.measure import CONFIGS, Setup
from tests.kit.benchmark.report import build_report, render_markdown
from tests.kit.benchmark.run import with_baseline
from tests.test_benchmark.test_benchmark_core import FLOPS, PEAK, _run, tiny_llama


def test_blocks_are_the_declared_layers_outermost_first():
    model = tiny_llama()
    blocks = transformer_blocks(model)
    assert [type(b).__name__ for b in blocks] == ["LlamaDecoderLayer"] * 2
    assert blocks[0] is model.model.layers[0]
    assert transformer_blocks(nn.Linear(2, 2)) == []


def test_baseline_config_has_no_plugin():
    config = CONFIGS["pytorch_fsdp2_bf16"]
    assert config.plugin_fn is None and config.wrap_fn is not None and config.dtype == "bf16"


def _report(config, step_time, reserved):
    run = _run([step_time], setup=asdict(Setup(entry="e", size="s", config=config, seq_len=16, micro_batch=2)))
    run["memory_bytes"] = {"reserved_peak": reserved}
    # complete enough to render
    flops = dict(FLOPS, analytic=1e9, counted=1e9)
    peak = dict(PEAK, nominal_source="vendor", measured_source="matmuls")
    return build_report([run], {}, {"commit": "abc"}, ["cmd"], flops, peak)


def test_comparison_with_the_baseline():
    ours, theirs = _report("zero2_bf16", 1.0, 40 << 30), _report("pytorch_fsdp2_bf16", 1.25, 32 << 30)
    result = with_baseline(copy.deepcopy(ours), theirs)
    assert result["baseline"]["throughput_ratio"] == pytest.approx(1.25)
    assert result["baseline"]["reserved_ratio"] == pytest.approx(1.25)
    markdown = render_markdown(result)
    assert "1.25× the baseline's throughput" in markdown and "pytorch_fsdp2_bf16 (baseline)" in markdown
    assert "- comparison with a baseline" not in markdown
    assert "Not compared by this report." in render_markdown(ours)

    other = _report("pytorch_fsdp2_bf16", 1.0, 1)
    other["setup"]["size"] = "t"
    with pytest.raises(ValueError, match="same setup"):
        with_baseline(copy.deepcopy(ours), other)
    fewer = copy.deepcopy(theirs)
    fewer["world_size"] = 1
    with pytest.raises(ValueError, match="as many devices"):
        with_baseline(copy.deepcopy(ours), fewer)
