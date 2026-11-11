"""Self-tests of data-parallel scaling: the communication shares of a trace, the efficiency curve, and a two-process
run on the CPU."""

import pytest

from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.benchmark.measure import STEP_ANNOTATION, Setup, communication_shares
from tests.kit.benchmark.report import render_markdown
from tests.kit.benchmark.scaling import run_scaling, scaling_curve
from tests.test_benchmark.test_benchmark_core import TINY


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
    assert communication_shares(events) == {"communication": 0.4, "exposed": 0.2}
    # outside every profiled step: not counted
    events.append(_kernel("ncclDevKernel_AllGather", 150, 190))
    assert communication_shares(events)["communication"] == 0.4
    # two steps: shares of their summed time
    events += [_step(6, 200, 300), _kernel("ncclDevKernel_AllReduce", 200, 300)]
    assert communication_shares(events) == {"communication": 140 / 200, "exposed": 120 / 200}
    with pytest.raises(ValueError):
        communication_shares([_kernel("x", 0, 1)])


def _report(devices, per_device, communication=None):
    return {
        "world_size": devices,
        "throughput": {
            "tokens_per_s": per_device * devices,
            "tokens_per_s_per_device": per_device,
            "step_time_s": {"median": 1.0},
            "mfu_nominal": 0.5,
        },
        "memory_bytes": {"reserved_peak": 1 << 30},
        "headline_repeat": 0,
        "repeats": [{"communication": communication}],
        "flags": [],
    }


def test_efficiency_is_per_device_throughput_against_the_smallest_count():
    reports = [
        _report(1, 1000),
        _report(2, 950, [{"communication": 0.1, "exposed": 0.02}, {"communication": 0.12, "exposed": 0.05}]),
        _report(8, 800, [None] * 8),
    ]
    points = scaling_curve(reports)
    assert [p["efficiency"] for p in points] == [1.0, 0.95, 0.8]
    assert points[1]["communication"] == 0.12 and points[1]["exposed_communication"] == 0.05
    assert points[0]["communication"] is None and points[2]["exposed_communication"] is None
    # a model that only fits on two devices: the base is two
    assert scaling_curve(reports[1:])[1]["efficiency"] == pytest.approx(800 / 950)


@rerun_if_address_is_in_use()
def test_cpu_scaling_end_to_end(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    setup = Setup(
        entry="tiny", size="tiny", config="ddp_fp32", seq_len=16, micro_batch=2, warmup_steps=1, timed_steps=2
    )
    result = run_scaling(setup, counts=(2, 1), repeats=1, spec=TINY)
    assert result["world_size"] == 2 and result["setup"]["profile_steps"] == 3
    assert [p["devices"] for p in result["scaling"]] == [1, 2]
    assert result["scaling"][0]["efficiency"] == 1.0
    # the CPU has no device profiler to read: no communication shares, and the report says n/a
    assert result["scaling"][1]["communication"] is None
    markdown = render_markdown(result)
    assert "## Data-parallel scaling" in markdown and "Not measured by this report." not in markdown
    assert "- data-parallel scaling efficiency" not in markdown
    with pytest.raises(ValueError):
        run_scaling(setup, counts=(0,), spec=TINY)
