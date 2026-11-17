"""Self-tests of data-parallel scaling: the device timeline of a trace, the communication shares, the efficiency curve,
and a two-process run on the CPU."""

import pytest

from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.benchmark.config.setup import Setup
from tests.kit.benchmark.metrics.parallel import communication_metrics, communication_shares
from tests.kit.benchmark.metrics.summary import scaling_curve, summarize
from tests.kit.benchmark.runner.timeline import STEP_ANNOTATION, spans_from_trace
from tests.kit.benchmark.scenarios.report import build_report, render_markdown
from tests.kit.benchmark.scenarios.scaling import run_scaling
from tests.test_benchmark.test_benchmark_core import FLOPS, PEAK, TINY, record


def _kernel(name, start, end):
    return {"cat": "kernel", "name": name, "ts": start, "dur": end - start}


def _step(index, start, end):
    return {"cat": "gpu_user_annotation", "name": f"{STEP_ANNOTATION}{index}", "ts": start, "dur": end - start}


def test_communication_shares_separate_hidden_from_exposed():
    events = [
        _step(5, 0, 100),
        _kernel("ampere_bf16_gemm", 0, 40),
        _kernel("ampere_bf16_gemm", 60, 100),
        # 40 of communication, of which 20 overlap the gemms: 20 exposed
        _kernel("ncclDevKernel_ReduceScatter_Sum_bf16", 30, 70),
        {"cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 100},
    ]
    spans = spans_from_trace(events)
    # the timeline keeps intervals, the gemms merged where they touch
    assert spans["steps"] == [(0, 100)] and spans["communication"] == [(30, 70)]
    assert communication_shares(spans) == {"communication": 0.4, "exposed": 0.2}
    # outside every profiled step: not counted
    events.append(_kernel("ncclDevKernel_AllGather", 150, 190))
    assert communication_shares(spans_from_trace(events))["communication"] == 0.4
    # two steps: shares of their summed time
    events += [_step(6, 200, 300), _kernel("ncclDevKernel_AllReduce", 200, 300)]
    assert communication_shares(spans_from_trace(events)) == {"communication": 140 / 200, "exposed": 120 / 200}
    with pytest.raises(ValueError):
        spans_from_trace([_kernel("x", 0, 1)])


def test_communication_is_the_largest_share_over_ranks():
    run = record([1.0])
    assert communication_metrics(run) is None
    quiet = {"steps": [(0, 100)], "communication": [(0, 10)], "compute": [(0, 100)]}
    busy = {"steps": [(0, 100)], "communication": [(0, 30)], "compute": [(10, 100)]}
    run["ranks"] = [dict(run["ranks"][0], timeline=quiet), dict(run["ranks"][1], timeline=busy)]
    assert communication_metrics(run) == {"communication": 0.3, "exposed": 0.1}


def _summary(devices, step_time):
    # 32 tokens per device per step: per device throughput 32 / step_time
    return summarize([record([step_time], ranks=devices)], FLOPS, PEAK)


def test_efficiency_is_per_device_throughput_against_the_smallest_count():
    summaries = [_summary(1, 1.0), _summary(2, 1 / 0.95), _summary(8, 1.25)]
    points = scaling_curve(summaries)
    assert [p["devices"] for p in points] == [1, 2, 8]
    assert [p["efficiency"] for p in points] == pytest.approx([1.0, 0.95, 0.8])
    assert points[0]["communication"] is None and points[2]["exposed_communication"] is None
    # a model that only fits on two devices: the base is two
    assert scaling_curve(summaries[1:])[1]["efficiency"] == pytest.approx(0.8 / 0.95)


@rerun_if_address_is_in_use()
def test_cpu_scaling_end_to_end(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    setup = Setup(
        entry="tiny", size="tiny", config="ddp_fp32", seq_len=16, micro_batch=2, warmup_steps=1, timed_steps=2
    )
    scaling = run_scaling(setup, counts=(2, 1), repeats=1, spec=TINY)
    measurement = scaling["measurements"][-1]
    assert measurement["summary"]["devices"] == 2 and measurement["setup"]["profile_steps"] == 3
    assert [p["devices"] for p in scaling["curve"]] == [1, 2]
    assert scaling["curve"][0]["efficiency"] == 1.0
    # the CPU has no device profiler to read: no communication shares, and the report says n/a
    assert scaling["curve"][1]["communication"] is None
    result = build_report(measurement, {}, [], scaling=scaling)
    assert len(result["scaling"]["measurements"]) == 1
    markdown = render_markdown(result)
    assert "## Data-parallel scaling" in markdown and "Not measured by this report." not in markdown
    assert "- data-parallel scaling efficiency" not in markdown
    with pytest.raises(ValueError):
        run_scaling(setup, counts=(0,), spec=TINY)
