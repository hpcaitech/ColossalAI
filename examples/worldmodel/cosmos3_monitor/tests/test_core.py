"""Unit tests for the monitor's read-only core: readers, discovery, state, alerts and parsing."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from examples.worldmodel.cosmos3_monitor.alerts import AlertEngine
from examples.worldmodel.cosmos3_monitor.discovery import discover, load_manifest, world_size
from examples.worldmodel.cosmos3_monitor.readers import JsonlTail
from examples.worldmodel.cosmos3_monitor.series import SeriesStore
from examples.worldmodel.cosmos3_monitor.state import RunState
from examples.worldmodel.cosmos3_monitor.system import HostSampler, parse_npu_smi


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_lines(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_run(tmp_path: Path, *, world: int = 2, steps: int = 3, completed: bool = True, fresh: bool = True):
    run = tmp_path / "run-a"
    metrics = run / "metrics"
    write_json(
        metrics / "run.json",
        {
            "run_id": "run-a",
            "run_name": "test",
            "world_size": world,
            "args": {"max_iter": 10},
            "started_at": time.time() - 100,
            "topology": [
                {"rank": rank, "hostname": "nodeA", "local_rank": rank, "device_index": rank} for rank in range(world)
            ],
        },
    )
    rows = []
    for step in range(1, steps + 1):
        rows.append(
            {
                "type": "record",
                "schema": 1,
                "step": step,
                "rank": 0,
                "time": time.time() - (steps - step) * 10,
                "ranks": list(range(world)),
                "world": world,
                "expected_world": world,
                "complete": True,
                "data": {
                    "samples": 8,
                    "ar_tokens": 1000,
                    "dm_tokens": 2000,
                    "wait_ms": 100.0,
                    "vae_ms": 200.0,
                    "epoch": 0,
                },
                "loss": {"total": 2.0 - step * 0.1, "vision": 1.0, "action": 0.5},
                "perf": {"step_ms": 1000.0 * step, "forward_ms": 300.0, "backward_ms": 500.0, "optim_ms": 50.0},
                "mem": {"device_used_gib": 40.0, "device_total_gib": 64.0, "peak_reserved_gib": 44.0},
                "optim": {"lr": 2e-4, "grad_norm": 3.0},
            }
        )
    write_lines(metrics / "aggregated.jsonl", rows)
    now = time.time() if fresh else time.time() - 3600
    for rank in range(world):
        write_json(
            metrics / f"heartbeat-{rank:03d}.json",
            {
                "schema": 1,
                "rank": rank,
                "pid": 1000 + rank,
                "hostname": "nodeA",
                "phase": "step_complete" if completed else "forward",
                "phase_time": now,
                "progress_time": now,
                "step": steps,
                "time": now,
                "memory": {"device_used_gib": 40.0, "device_total_gib": 64.0},
            },
        )
        if completed:
            write_json(
                metrics / f"terminal-{rank:03d}.json", {"rank": rank, "step": steps, "completed": True, "time": now}
            )
    return run


def test_jsonl_tail_keeps_a_partial_line_and_resets_after_truncation(tmp_path):
    path = tmp_path / "rank-000.jsonl"
    path.write_text('{"step": 1}\n{"step":', encoding="utf-8")
    tail = JsonlTail(path)
    assert [row["step"] for row in tail.read_new()] == [1]
    with path.open("a", encoding="utf-8") as stream:
        stream.write(" 2}\n")
    assert [row["step"] for row in tail.read_new()] == [2]
    path.write_text('{"step": 9}\n', encoding="utf-8")
    assert [row["step"] for row in tail.read_new()] == [9]
    assert tail.truncated is True


def test_discovery_understands_all_three_layouts(tmp_path):
    make_run(tmp_path)
    validation = tmp_path / "validation"
    write_json(validation / "config.json", {"run_id": "v", "nodes": ["node1"], "devices": [0, 1], "args": []})
    write_lines(validation / "metrics" / "aggregated.jsonl", [{"type": "record", "step": 1}])
    legacy = tmp_path / "legacy"
    write_json(legacy / "run.json", {"run_id": "l", "world_size": 1})
    (legacy / "metrics").mkdir()
    refs = discover([tmp_path], depth=3)
    layouts = {ref.layout for ref in refs}
    assert {"metrics", "validation", "legacy"} <= layouts
    assert len({ref.token for ref in refs}) == len(refs)


def test_world_size_comes_from_manifest_launcher_or_config(tmp_path):
    run = make_run(tmp_path, world=4)
    ref = discover([run])[0]
    manifest = load_manifest(ref)
    assert world_size(manifest) == 4
    assert world_size({"config": {"nodes": ["a", "b"], "devices": [0, 1, 2]}}) == 6


def test_state_machine_completes_a_clean_run(tmp_path):
    run = make_run(tmp_path, world=2, steps=3, completed=True)
    ref = discover([run])[0]
    state = RunState(ref)
    state.refresh()
    snapshot = state.snapshot()
    assert snapshot["state"] == "completed"
    assert snapshot["step"] == 3
    assert snapshot["world"] == 2
    assert snapshot["live_ranks"] == 2
    assert snapshot["progress"] == pytest.approx(0.3)
    assert snapshot["last"]["loss"] == pytest.approx(1.7)
    assert snapshot["last"]["samples_per_s"] == pytest.approx(8 / 3.0)
    assert state.issues() == []


def test_state_machine_flags_stalled_progress(tmp_path):
    run = make_run(tmp_path, world=1, steps=1, completed=False, fresh=True)
    metrics = run / "metrics"
    heartbeat = json.loads((metrics / "heartbeat-000.json").read_text())
    heartbeat["time"] = time.time()
    heartbeat["progress_time"] = time.time() - 4000
    heartbeat["phase_time"] = time.time() - 4000
    write_json(metrics / "heartbeat-000.json", heartbeat)
    ref = discover([run])[0]
    state = RunState(ref)
    state.refresh()
    kinds = {issue["kind"] for issue in state.issues()}
    assert "stalled_progress" in kinds
    assert state.state() == "attention"


def test_alert_engine_deduplicates_and_expires_transient_events(tmp_path):
    engine = AlertEngine(tmp_path / "state")
    now = time.time()
    event = {
        "type": "event",
        "severity": "error",
        "kind": "nonfinite_loss",
        "step": 5,
        "rank": 0,
        "time": now,
        "message": "NaN",
    }
    issues = [{"kind": "stalled_progress", "severity": "critical", "rank": 0, "message": "stalled"}]
    first = engine.evaluate("tok", "run", issues, [event], now=now)
    second = engine.evaluate("tok", "run", issues, [event], now=now + 1)
    assert len(first) == 2 and len(second) == 2
    stalled = next(item for item in second if item["rule"] == "stalled_progress")
    assert stalled["count"] == 2
    resolved = engine.evaluate("tok", "run", [], [], now=now + 700)
    assert resolved == []
    assert engine.history("tok")
    # An event from an hour ago is history, not a current alert.
    old = dict(event, time=now - 3600)
    assert engine.evaluate("other", "run", [], [old], now=now) == []


def test_series_decimation_keeps_extremes_and_partial_points():
    store = SeriesStore()
    for step in range(1, 1001):
        store.append(
            {
                "type": "record",
                "step": step,
                "world": 8,
                "complete": step != 500,
                "data": {"samples": 8},
                "loss": {"total": 1000.0 if step == 500 else 1.0},
                "perf": {"step_ms": 1000.0},
                "mem": {"device_used_gib": 40.0, "device_total_gib": 64.0},
            }
        )
    raw = store.query(["loss/total", "perf/samples_per_s"], max_points=2000)
    assert raw["perf/samples_per_s"]["y"][499] is None  # the partial step has no throughput
    result = store.query(["loss/total"], max_points=100)
    xs = result["loss/total"]["x"]
    ys = result["loss/total"]["y"]
    assert len(xs) <= 2 * 100 + 1
    assert max(y for y in ys if y is not None) == 1000.0


NPU_SMI = """
+------------------------------------------------------------------------------------------------+
| npu-smi 25.2.1                   Version: 25.2.1                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B2               | OK            | 92.3        36                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          9868 / 65536         |
+===========================+===============+====================================================+
| 1     910B2               | OK            | 91.8        37                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          3427 / 65536         |
+===========================+===============+====================================================+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 0       0                 | 63111         |                          | 6486                    |
+===========================+===============+====================================================+
| No running processes found in NPU 1                                                            |
+===========================+===============+====================================================+
"""


def test_parse_npu_smi_reads_hbm_temperature_and_processes():
    parsed = parse_npu_smi(NPU_SMI)
    assert [device["index"] for device in parsed["devices"]] == [0, 1]
    first = parsed["devices"][0]
    assert first["hbm_used_mib"] == 9868
    assert first["hbm_total_mib"] == 65536
    assert first["temperature_c"] == 36
    assert first["power_w"] == 92.3
    assert first["processes"][0]["pid"] == 63111
    assert "aicore_pct" not in first
    assert parsed["devices"][1]["processes"] == []


def test_sampler_never_exposes_aicore_as_a_metric(tmp_path):
    sampler = HostSampler(disk_paths=[str(tmp_path)])
    payload = sampler.sample()
    text = json.dumps(payload).lower().replace("aicore_utilization_excluded", "")
    assert "aicore" not in text
    assert payload["advisory"]["aicore_utilization_excluded"] is True


def test_remote_agent_and_mirror_round_trip(tmp_path):
    from examples.worldmodel.cosmos3_monitor.mirror import Mirror
    from examples.worldmodel.cosmos3_monitor.remote_agent import collect

    metrics = tmp_path / "remote" / "metrics"
    control = tmp_path / "remote" / "control"
    metrics.mkdir(parents=True)
    control.mkdir(parents=True)
    write_json(metrics / "run.json", {"run_id": "mirror-run", "world_size": 1})
    write_lines(
        metrics / "aggregated.jsonl",
        [
            {"type": "record", "step": 1, "rank": 0, "world": 1, "data": {"samples": 1}, "loss": {"total": 1.0}},
            {"type": "event", "step": 1, "rank": 0, "severity": "warn", "kind": "x", "message": "m"},
        ],
    )
    write_json(metrics / "heartbeat-000.json", {"rank": 0, "phase": "data", "step": 1, "time": time.time()})
    write_json(control / "launcher-0.json", {"state": "running", "node": "nodeA"})
    payload = collect(metrics, control, {}, 0, "no-such-entry", "")
    assert payload["aggregated"]["size"] > 0
    assert payload["files"]["heartbeat-000.json"]["phase"] == "data"
    assert payload["launcher"]["state"] == "running"

    mirror = Mirror(
        {"run_id": "mirror-run", "out": str(tmp_path / "remote"), "nodes": ["nodeA"]},
        cache_root=tmp_path / "cache",
        remote_repo="/repo",
        metrics_dir=str(metrics),
        control_dir=str(control),
    )
    mirror._ingest("nodeA", payload)
    assert (tmp_path / "cache" / "mirror-run" / "metrics" / "aggregated.jsonl").read_text().count("\n") == 2
    assert (tmp_path / "cache" / "mirror-run" / "metrics" / "run.json").is_file()
    # A second identical poll must not duplicate the aggregated bytes.
    payload2 = collect(metrics, control, {"aggregated": payload["aggregated"]["offset"]}, 0, "no-such-entry", "")
    mirror._ingest("nodeA", payload2)
    assert (tmp_path / "cache" / "mirror-run" / "metrics" / "aggregated.jsonl").read_text().count("\n") == 2


def test_step_detail_reports_each_stage_and_mfu(tmp_path):
    run = make_run(tmp_path, world=2, steps=3, completed=True)
    ref = discover([run])[0]
    state = RunState(ref)
    state.refresh()
    detail = state.step_detail(2)
    assert detail is not None
    assert detail["iteration"] == 2
    assert detail["epoch"] == 0
    assert detail["batch_samples"] == 8
    stages = {stage["name"]: stage for stage in detail["stages"]}
    assert {"data wait", "VAE", "forward", "backward", "optimizer"} <= set(stages)
    assert stages["forward"]["seconds"] is not None and stages["forward"]["seconds"] > 0
    assert stages["forward"]["mfu"] is not None and stages["forward"]["mfu"] > 0
    # Activation checkpointing is the default: backward carries 3x the forward FLOPs.
    assert stages["backward"]["flops"] == pytest.approx(stages["forward"]["flops"] * 3)
    assert detail["total_mfu"] is not None and detail["total_mfu"] > 0
    assert state.step_detail(99) is None
