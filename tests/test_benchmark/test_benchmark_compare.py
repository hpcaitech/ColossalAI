"""Self-tests of comparisons: the plain-PyTorch FSDP2 baseline's sharding, the comparison's ratios and report, and a
comparison of two implementations on the CPU."""

from dataclasses import asdict

import pytest
from torch import nn

from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.benchmark.config.model import RANDOM
from tests.kit.benchmark.config.plugins import CONFIGS, transformer_blocks
from tests.kit.benchmark.config.setup import Setup
from tests.kit.benchmark.metrics.summary import comparison, summarize
from tests.kit.benchmark.scenarios.compare import run_compare
from tests.kit.benchmark.scenarios.report import against, build_report, render_markdown
from tests.kit.benchmark.scenarios.single import run_single
from tests.test_benchmark.test_benchmark_core import FLOPS, PEAK, TINY, record, tiny_llama


def test_blocks_are_the_declared_layers_outermost_first():
    model = tiny_llama()
    blocks = transformer_blocks(model)
    assert [type(b).__name__ for b in blocks] == ["LlamaDecoderLayer"] * 2
    assert blocks[0] is model.model.layers[0]
    assert transformer_blocks(nn.Linear(2, 2)) == []


def test_offload_configs_run_adamw_as_hybrid_adam():
    from tests.kit.benchmark.config.optimizers import HYBRID_ADAM, for_cpu

    offload = {name for name, config in CONFIGS.items() if config.cpu_optimizer}
    assert offload == {"zero1_bf16_offload", "zero2_bf16_offload"}
    assert for_cpu("adamw") == HYBRID_ADAM and for_cpu("muon") == "muon"


def test_baseline_config_has_no_plugin():
    config = CONFIGS["pytorch_fsdp2_bf16"]
    assert config.plugin_fn is None and config.wrap_fn is not None and config.dtype == "bf16"


def _measurement(config, step_time, reserved, device=None):
    setup = asdict(Setup(entry="e", size="s", config=config, seq_len=16, micro_batch=2))
    run = record([step_time], setup=setup, memory={"reserved_peak": reserved})
    if device is not None:
        run["environment"] = {"accelerator": "x", "device_name": device}
    flops = dict(FLOPS, analytic=1e9, counted=1e9, counted_implementation="default")
    peak = dict(PEAK, nominal_source="vendor", measured_source="matmuls")
    return {"setup": setup, "flops": flops, "peak": peak, "summary": summarize([run], flops, peak), "runs": [run]}


def test_comparison_with_the_baseline():
    ours, theirs = _measurement("zero2_bf16", 1.0, 40 << 30), _measurement("pytorch_fsdp2_bf16", 1.25, 32 << 30)
    ratios = comparison(ours["summary"], theirs["summary"])
    assert ratios == {"throughput_ratio": pytest.approx(1.25), "reserved_ratio": pytest.approx(1.25)}
    compared = {"field": "config", "ours": "zero2_bf16", "theirs": "pytorch_fsdp2_bf16", "measurement": theirs}
    markdown = render_markdown(build_report(ours, {"commit": "abc"}, ["cmd"], comparison=dict(compared, ratios=ratios)))
    assert "differ only in `config`" in markdown
    assert "zero2_bf16 trains at 1.25× the throughput of pytorch_fsdp2_bf16" in markdown
    assert "| pytorch_fsdp2_bf16 | 26 |" in markdown  # 32 tokens / 1.25 s per device
    assert "- comparison with another setup" not in markdown
    assert "Not compared by this report." in render_markdown(build_report(ours, {}, []))


def test_comparison_changes_one_comparable_field():
    ours = _measurement("zero2_bf16", 1.0, 1)
    with pytest.raises(ValueError, match="cannot compare"):
        run_compare(ours, "entry", "other")
    with pytest.raises(ValueError, match="cannot compare"):
        run_compare(ours, "hidden_size", 1)
    with pytest.raises(ValueError, match="both measurements"):
        run_compare(ours, "config", "zero2_bf16")


@rerun_if_address_is_in_use()
def test_cpu_comparison_of_two_implementations(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    setup = Setup(
        entry="tiny",
        size="tiny",
        config="ddp_fp32",
        seq_len=16,
        micro_batch=2,
        warmup_steps=1,
        timed_steps=2,
        weights=RANDOM,
    )
    ours = run_single(setup, repeats=1, spec=TINY)
    compared = run_compare(ours, "implementation", "unchanged", repeats=1, spec=TINY)
    assert compared["measurement"]["setup"] == dict(ours["setup"], implementation="unchanged")
    assert compared["ratios"]["throughput_ratio"] > 0
    markdown = render_markdown(build_report(ours, {}, [], comparison=compared))
    assert "differ only in `implementation`" in markdown and "| unchanged |" in markdown


def test_two_machines_compare_through_their_reports():
    h20 = build_report(_measurement("zero2_bf16", 1.0, 40 << 30, device="NVIDIA H20"), {"commit": "a" * 40}, ["cmd"])
    npu = build_report(_measurement("zero2_bf16", 2.0, 30 << 30, device="Ascend910B2"), {"commit": "a" * 40}, ["cmd"])
    both = against(h20, npu)
    assert both["comparison"]["ratios"]["throughput_ratio"] == pytest.approx(2.0)
    markdown = render_markdown(both)
    assert (
        "differ only in `machine`" in markdown
        and "NVIDIA H20 trains at 2.00× the throughput of Ascend910B2" in markdown
    )
    assert "MFU of the measured peak" in markdown and "different commits" not in markdown
    later = build_report(_measurement("zero2_bf16", 2.0, 30 << 30, device="Ascend910B2"), {"commit": "b" * 40}, [])
    assert "different commits" in render_markdown(against(h20, later))

    other = build_report(_measurement("gemini_bf16", 2.0, 1, device="Ascend910B2"), {}, [])
    with pytest.raises(ValueError, match="differing in \\['config'\\]"):
        against(h20, other)
    with pytest.raises(ValueError, match="already has a comparison"):
        against(both, npu)
