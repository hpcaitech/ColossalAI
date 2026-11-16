"""Unit tests for the monitor's read-only core: readers, discovery, state, alerts and parsing."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from unittest import mock

import pytest

from examples.worldmodel.cosmos3_monitor.alerts import AlertEngine
from examples.worldmodel.cosmos3_monitor.discovery import discover, load_manifest, world_size
from examples.worldmodel.cosmos3_monitor.readers import JsonlTail
from examples.worldmodel.cosmos3_monitor.series import SeriesStore, normalise_loss_record
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
                "perf": {
                    "step_ms": 1000.0 * step,
                    "forward_ms": 300.0,
                    "backward_ms": 500.0,
                    "optim_ms": 50.0,
                    "flops": 1.2e14,
                    "flops_forward": 2.5e13,
                    "flops_backward": 7.5e13,
                    "flops_vae": 2.0e13,
                    "mfu": 0.35,
                    "mfu_forward": 0.55,
                    "mfu_backward": 0.25,
                    "mfu_vae": 0.80,
                    "peak_tflops_per_npu": 353.9,
                    "checkpointing_mode": "full",
                },
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
    # A shrink may be a copy in progress, so the first short poll waits; a file that stays short is rebuilt.
    assert tail.read_new() == []
    tail.shrink_grace_s = 0.0
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


def test_mirror_merges_every_node_and_preserves_heartbeat_age(tmp_path):
    from examples.worldmodel.cosmos3_monitor.mirror import Mirror
    from examples.worldmodel.cosmos3_monitor.remote_agent import collect

    root = tmp_path / "remote"
    metrics_a = root / "a" / "metrics"
    metrics_b = root / "b" / "metrics"
    control = root / "control"
    for directory in (metrics_a, metrics_b, control):
        directory.mkdir(parents=True)
    write_json(metrics_a / "run.json", {"run_id": "multi", "world_size": 16})
    write_lines(metrics_a / "aggregated.jsonl", [{"type": "record", "step": 1, "rank": 0}])
    write_lines(metrics_b / "aggregated.jsonl", [{"type": "record", "step": 2, "rank": 8}])
    old = time.time() - 10
    for metrics, rank in ((metrics_a, 0), (metrics_b, 8)):
        heartbeat = metrics / f"heartbeat-{rank:03d}.json"
        write_json(heartbeat, {"rank": rank, "phase": "forward", "step": 1, "time": old})
        os.utime(heartbeat, (old, old))
        write_lines(
            metrics / f"rank-{rank:03d}.jsonl",
            [{"type": "record", "step": 1, "rank": rank, "data": {"samples": 4}, "perf": {"step_ms": 1000}}],
        )
    write_lines(metrics_a / "system" / "node-A.jsonl", [{"hostname": "A", "time": old, "devices": []}])
    write_lines(metrics_b / "system" / "node-B.jsonl", [{"hostname": "B", "time": old, "devices": []}])

    mirror = Mirror(
        {"run_id": "multi", "out": str(root / "out")},
        cache_root=tmp_path / "cache",
        nodes=["nodeA", "nodeB"],
        remote_repo="/repo",
        metrics_dir=str(metrics_a),
        control_dir=str(control),
    )
    with mock.patch("examples.worldmodel.cosmos3_monitor.remote_agent.socket.gethostname", return_value="A"):
        payload_a = collect(metrics_a, control, {}, 0, "no-such-entry", "")
    with mock.patch("examples.worldmodel.cosmos3_monitor.remote_agent.socket.gethostname", return_value="B"):
        payload_b = collect(metrics_b, control, {}, 1, "no-such-entry", "")
    mirror._ingest("nodeA", payload_a)
    mirror._ingest("nodeB", payload_b)
    cache = tmp_path / "cache" / "multi" / "metrics"
    assert (cache / "rank-000.jsonl").is_file()
    assert (cache / "rank-008.jsonl").is_file()
    assert (cache / "heartbeat-000.json").is_file()
    assert (cache / "heartbeat-008.json").is_file()
    assert (cache / "system" / "node-A.jsonl").is_file()
    assert (cache / "system" / "node-B.jsonl").is_file()
    # Only rank 0's aggregated ledger is authoritative; a non-primary node may hold a stale slice.
    assert '"rank": 0' in (cache / "aggregated.jsonl").read_text()
    assert '"rank": 8' not in (cache / "aggregated.jsonl").read_text()
    # The mirror preserves the remote heartbeat age instead of stamping the copy with "now".
    age = time.time() - (cache / "heartbeat-008.json").stat().st_mtime
    assert 8 <= age <= 13


def test_remote_agent_only_mirrors_its_own_system_file(tmp_path):
    from examples.worldmodel.cosmos3_monitor.remote_agent import collect

    metrics = tmp_path / "remote" / "metrics"
    control = tmp_path / "remote" / "control"
    control.mkdir(parents=True)
    own = f"node-{socket.gethostname()}.jsonl"
    write_lines(metrics / "system" / own, [{"hostname": socket.gethostname(), "time": time.time(), "devices": []}])
    # A copied file from another node must not be mirrored a second time through this node's offsets.
    write_lines(metrics / "system" / "node-other.jsonl", [{"hostname": "other", "time": time.time(), "devices": []}])
    payload = collect(metrics, control, {}, 0, "no-such-entry", "")
    assert set(payload["system"]) == {own}


def test_snapshot_reports_every_host_even_when_one_has_no_heartbeat(tmp_path):
    run = tmp_path / "run-coverage"
    metrics = run / "metrics"
    metrics.mkdir(parents=True)
    write_json(
        metrics / "run.json",
        {
            "run_id": "coverage",
            "world_size": 4,
            "topology": [
                {"rank": rank, "hostname": "A" if rank < 2 else "B", "local_rank": rank % 2, "device_index": rank % 2}
                for rank in range(4)
            ],
        },
    )
    now = time.time()
    for rank in (0, 1):
        write_json(
            metrics / f"heartbeat-{rank:03d}.json",
            {"rank": rank, "hostname": "A", "phase": "forward", "step": 3, "time": now, "local_rank": rank},
        )
    state = RunState(discover([run])[0])
    state.refresh()
    snapshot = state.snapshot()
    coverage = snapshot["coverage"]
    assert coverage["nodes_expected"] == 2
    assert coverage["nodes_reporting"] == 1
    assert coverage["nodes_live"] == 1
    assert snapshot["live_ranks"] == 2
    hosts = {row["hostname"]: row for row in coverage["hosts"]}
    assert hosts["B"]["live"] == 0
    assert hosts["B"]["missing"] == 2
    ranks = state.ranks_view()
    assert ranks["ranks"][0]["hostname"] == "A"
    assert ranks["ranks"][2]["status"] == "missing"


def test_partial_aggregate_uses_the_global_rank_throughput(tmp_path):
    run = tmp_path / "run-partial"
    metrics = run / "metrics"
    metrics.mkdir(parents=True)
    now = time.time()
    write_json(
        metrics / "run.json",
        {
            "run_id": "partial",
            "world_size": 4,
            "started_at": now - 100,
            "args": {"max_iter": 10},
            "topology": [
                {"rank": rank, "hostname": "A", "local_rank": rank, "device_index": rank} for rank in range(4)
            ],
        },
    )
    # Rank 0's aggregate saw only half the ranks; its samples and throughput are deliberately partial.
    write_lines(
        metrics / "aggregated.jsonl",
        [
            {
                "type": "record",
                "step": 1,
                "rank": 0,
                "time": now - 5,
                "ranks": [0, 1, 2, 3],
                "world": 4,
                "expected_world": 4,
                "complete": False,
                "data": {"samples": 8},
                "loss": {"total": 2.0},
                "perf": {
                    "step_ms": 1000.0,
                    "forward_ms": 300.0,
                    "backward_ms": 500.0,
                    "flops": 1.2e14,
                    "flops_forward": 2.5e13,
                    "flops_backward": 7.5e13,
                    "flops_vae": 0.0,
                    "mfu": 0.30,
                    "mfu_forward": 0.50,
                    "mfu_backward": 0.20,
                    "peak_tflops_per_npu": 353.9,
                    "checkpointing_mode": "full",
                },
            }
        ],
    )
    for rank in range(4):
        write_lines(
            metrics / f"rank-{rank:03d}.jsonl",
            [
                {
                    "type": "record",
                    "step": 1,
                    "rank": rank,
                    "time": now - 5,
                    "data": {"samples": 4},
                    "perf": {
                        "step_ms": 1000.0,
                        "flops": 3.0e13,
                        "flops_forward": 6.25e12,
                        "flops_backward": 1.875e13,
                        "mfu": 0.35,
                        "mfu_forward": 0.55,
                        "mfu_backward": 0.25,
                        "peak_tflops_per_npu": 353.9,
                        "checkpointing_mode": "full",
                    },
                    "loss": {"total": 2.0},
                }
            ],
        )
        write_json(
            metrics / f"heartbeat-{rank:03d}.json",
            {"rank": rank, "hostname": "A", "phase": "step_complete", "step": 1, "time": now, "local_rank": rank},
        )
    state = RunState(discover([run])[0])
    state.refresh()
    snapshot = state.snapshot()
    assert snapshot["last"]["samples"] == 16
    assert snapshot["last"]["samples_per_s"] == 16
    assert snapshot["last"]["mfu"] == pytest.approx(0.35)
    assert snapshot["eta_s"] is not None and 8 <= snapshot["eta_s"] <= 10
    assert state.series.records[-1]["data"]["samples"] == 16
    assert state.series.latest_value("data/samples") == 16
    assert state.series.latest_value("perf/samples_per_s") == 16
    detail = state.step_detail(1)
    assert detail is not None
    assert detail["batch_samples"] == 16
    assert detail["aggregate_from_ranks"] is True
    stages = {stage["name"]: stage for stage in detail["stages"]}
    assert stages["forward"]["seconds"] is not None


def test_partial_aggregate_scales_to_the_expected_world_before_rank_files_arrive(tmp_path):
    run = tmp_path / "run-estimated"
    metrics = run / "metrics"
    metrics.mkdir(parents=True)
    now = time.time()
    write_json(
        metrics / "run.json",
        {
            "run_id": "estimated",
            "world_size": 4,
            "started_at": now - 100,
            "args": {"max_iter": 10, "samples_per_rank": 4, "grad_accum": 1},
            "topology": [
                {"rank": rank, "hostname": "A", "local_rank": rank, "device_index": rank} for rank in range(4)
            ],
        },
    )
    write_lines(
        metrics / "aggregated.jsonl",
        [
            {
                "type": "record",
                "step": 1,
                "rank": 0,
                "time": now - 5,
                "ranks": [0, 1],
                "world": 2,
                "expected_world": 4,
                "complete": False,
                "data": {"partial_samples": 8, "ar_tokens": 100, "dm_tokens": 200},
                "loss": {"total": 2.0},
                "perf": {
                    "step_ms": 1000.0,
                    "flops": 1.2e14,
                    "flops_forward": 2.5e13,
                    "flops_backward": 7.5e13,
                    "mfu": 0.30,
                    "mfu_forward": 0.50,
                    "mfu_backward": 0.20,
                    "peak_tflops_per_npu": 353.9,
                    "checkpointing_mode": "full",
                },
            }
        ],
    )
    state = RunState(discover([run])[0])
    state.refresh()
    snapshot = state.snapshot()
    assert snapshot["last"]["samples"] == 16
    assert snapshot["last"]["samples_per_s"] == 16
    assert snapshot["last"]["mfu"] is not None
    record = state.series.records[-1]
    assert record["samples_estimated"] is True
    assert record["reporting_world"] == 2
    assert record["data"]["samples"] == 16
    assert record["data"]["partial_samples"] == 8
    assert record["data"]["ar_tokens"] == 200
    assert record["data"]["dm_tokens"] == 400
    assert record["perf"]["mfu"] == 0.30


def test_step_detail_does_not_guess_mfu_without_recorded_flops():
    from examples.worldmodel.cosmos3_monitor.series import stage_metrics

    record = {
        "step": 1,
        "data": {"samples": 8, "vae_ms": 200.0},
        "perf": {"step_ms": 1000.0, "forward_ms": 300.0, "backward_ms": 500.0},
    }
    detail = stage_metrics(record, 2)
    assert detail["total_mfu"] is None
    assert all(stage["mfu"] is None for stage in detail["stages"])
    assert detail["total_flops"] is None


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


def test_throughput_store_reports_rank_and_node_rates():
    from examples.worldmodel.cosmos3_monitor.series import ThroughputStore

    store = ThroughputStore()
    now = time.time()
    store.ingest(0, "nodeA", 1, 8, 1.0, now)
    store.ingest(1, "nodeA", 1, 8, 1.2, now)
    store.ingest(2, "nodeB", 1, 8, 1.5, now)
    host_ranks = {"nodeA": [0, 1], "nodeB": [2]}
    store.finalize_pending(host_ranks, 3, now)
    view = store.step_view(1, host_ranks)
    nodes = {node["host"]: node for node in view["nodes"]}
    assert nodes["nodeA"]["samples"] == 16
    assert nodes["nodeA"]["step_s"] == 1.2
    assert nodes["nodeA"]["throughput"] == pytest.approx(16 / 1.2)
    assert nodes["nodeB"]["throughput"] == pytest.approx(8 / 1.5)
    assert view["total"] == pytest.approx(24 / 1.5)
    assert len(view["ranks"]) == 3
    series = store.series("nodes", max_points=100)
    assert set(series) == {"nodeA", "nodeB"}


def test_step_detail_includes_per_rank_throughput(tmp_path):
    run = make_run(tmp_path, world=2, steps=3, completed=True)
    metrics = run / "metrics"
    for rank in range(2):
        rows = [
            {
                "type": "record",
                "schema": 1,
                "step": step,
                "rank": rank,
                "time": time.time(),
                "data": {"samples": 4, "epoch": 0},
                "perf": {"step_ms": 1000.0 + rank * 100},
            }
            for step in range(1, 4)
        ]
        write_lines(metrics / f"rank-{rank:03d}.jsonl", rows)
    ref = discover([run])[0]
    state = RunState(ref)
    state.refresh()
    detail = state.step_detail(2)
    throughput = detail["throughput"]
    assert len(throughput["ranks"]) == 2
    assert throughput["nodes"][0]["host"] == "nodeA"
    assert throughput["nodes"][0]["samples"] == 8
    assert throughput["nodes"][0]["step_s"] == pytest.approx(1.1)
    assert throughput["total"] == pytest.approx(8 / 1.1)


def test_running_run_reports_throughput_based_eta(tmp_path):
    run = make_run(tmp_path, world=1, steps=3, completed=False, fresh=True)
    metrics = run / "metrics"
    # Remove the partial-step marker and let the run look live: heartbeats exist, no terminal files.
    for path in metrics.glob("terminal-*.json"):
        path.unlink()
    manifest = json.loads((metrics / "run.json").read_text())
    manifest["args"]["max_iter"] = 10
    write_json(metrics / "run.json", manifest)
    ref = discover([run])[0]
    state = RunState(ref)
    state.refresh()
    snapshot = state.snapshot()
    assert snapshot["eta_s"] is not None and snapshot["eta_s"] > 0
    assert snapshot["eta_source"] == "throughput"
    assert snapshot["elapsed_s"] is not None
    assert snapshot["total_s"] == pytest.approx(snapshot["elapsed_s"] + snapshot["eta_s"])


def test_system_snapshot_keeps_process_tables_and_series_opt_in():
    """The dashboard's default system view must stay small: the process rows and the device series are
    only drawn by the System tab, and together they were ~90% of a four-node response."""
    from examples.worldmodel.cosmos3_monitor.system import SystemHistory

    history = SystemHistory()
    base = {
        "hostname": "A",
        "training_processes": 1,
        "processes": [{"pid": 7, "command": "train" * 50}],
        "ranked_processes": [{"pid": 7, "command": "train" * 50}],
        "devices": [
            {"index": 0, "hbm_used_mib": 100.0, "temperature_c": 40.0, "power_w": 300.0, "processes": [{"pid": 7}]}
        ],
    }
    now = time.time()
    for index in range(20):
        history.ingest({**base, "time": now - 20 + index})

    compact = history.snapshot()["hosts"]["A"]
    assert "processes" not in compact["latest"] and "ranked_processes" not in compact["latest"]
    assert compact["latest"]["process_count"] == 1
    assert compact["latest"]["devices"][0]["process_count"] == 1
    assert "processes" not in compact["latest"]["devices"][0]
    assert compact["devices"]["0"]["hbm_used_mib"] == []
    # The compact form still carries what the topology view draws.
    assert compact["latest"]["devices"][0]["hbm_used_mib"] == 100.0

    full = history.snapshot(processes=True, points=5)["hosts"]["A"]
    assert full["latest"]["processes"][0]["pid"] == 7
    assert len(full["devices"]["0"]["hbm_used_mib"]) == 5
    # The stored history is untouched: a later compact read is still compact.
    assert "processes" not in history.snapshot()["hosts"]["A"]["latest"]


def test_jsonl_tail_survives_an_atomic_replace_and_a_transient_short_copy(tmp_path):
    """The relay publishes by rename, and a plain copy is briefly short: neither is a rewritten history."""
    path = tmp_path / "rank-000.jsonl"
    write_lines(path, [{"step": 1}, {"step": 2}, {"step": 3}])
    tail = JsonlTail(path)
    assert [row["step"] for row in tail.read_new()] == [1, 2, 3]

    # Atomic replace: same path, new inode, one appended record.  Only the new record may come back.
    staging = tmp_path / ".rank-000.jsonl.tmp"
    write_lines(staging, [{"step": 1}, {"step": 2}, {"step": 3}, {"step": 4}])
    os.replace(staging, path)
    assert [row["step"] for row in tail.read_new()] == [4]

    # A copy caught mid-write: the file is shorter for a moment.  Nothing is returned and nothing is reset.
    path.write_text('{"step": 1}\n')
    assert tail.read_new() == []
    assert tail.truncated is False and tail.offset > 0
    # Once the copy finishes, reading resumes from the offset without duplicating the earlier records.
    write_lines(path, [{"step": 1}, {"step": 2}, {"step": 3}, {"step": 4}, {"step": 5}])
    assert [row["step"] for row in tail.read_new()] == [5]

    # A file that stays short is a real rewrite: after the grace period the tail starts over.
    path.write_text('{"step": 40}\n')
    assert tail.read_new() == []  # first short poll: wait, it may be a copy in progress
    tail.shrink_grace_s = 0.0  # the deadline passed
    assert [row["step"] for row in tail.read_new()] == [40]
    assert tail.truncated is True


def test_throughput_store_does_not_duplicate_points_on_a_re_read(tmp_path):
    from examples.worldmodel.cosmos3_monitor.series import ThroughputStore

    store = ThroughputStore()
    now = time.time()
    for step in (5, 6, 7):
        store.ingest(0, "h", step, 32, 1.0, now)
    # A re-read of the file from the start repeats steps 5-7; the series must not grow.
    for step in (5, 6, 7):
        store.ingest(0, "h", step, 32, 1.0, now)
    assert [int(x) for x, _ in store.rank_points[0]] == [5, 6, 7]
    assert sorted(store.step_ranks) == [5, 6, 7]


def test_legacy_droid_loss_conversion_is_explicit_idempotent_and_read_only():
    manifest = {"loss_objective": "droid_sample_modality_mean", "args": {"chunk_length": 32}}
    record = {"loss": {"total": 1.82, "vision": 1.7, "action": 0.12}}
    converted = normalise_loss_record(record, manifest)
    assert converted["loss"] == pytest.approx(
        {"total": 0.19125 + 0.012375, "vision": 0.19125, "action": 0.012375, "objective": 1.82}
    )
    assert record["loss"]["vision"] == 1.7
    assert normalise_loss_record(converted, manifest) is converted
    assert normalise_loss_record(record, {}) is record
    assert normalise_loss_record(record, {"loss_objective": "droid_sample_modality_mean"}) is record


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("partial_aggregate", [False, True])
def test_global_loss_is_corrected_after_delayed_rank_arrival(tmp_path, legacy, partial_aggregate):
    run = make_run(tmp_path, world=2, steps=1, completed=False)
    metrics = run / "metrics"
    manifest = json.loads((metrics / "run.json").read_text())
    manifest.update(loss_objective="droid_sample_modality_mean", args={"chunk_length": 32})
    write_json(metrics / "run.json", manifest)

    def rank_record(rank, samples, vision, action):
        objective = vision * 10 * 8 / 9 + action * 10 * 32 / 33
        loss = {"total": vision + action, "vision": vision, "action": action, "objective": objective}
        if legacy:
            loss = {"total": objective, "vision": vision * 10 * 8 / 9, "action": action * 10 * 32 / 33}
        return {
            "type": "record",
            "step": 1,
            "rank": rank,
            "time": time.time(),
            "data": {"samples": samples},
            "perf": {"step_ms": 1000},
            "loss": loss,
        }

    first, second = rank_record(0, 1, 1.0, 4.0), rank_record(1, 3, 9.0, 16.0)
    if partial_aggregate:
        write_lines(
            metrics / "aggregated.jsonl", [{**first, "ranks": [0], "world": 1, "expected_world": 2, "complete": False}]
        )
    else:
        (metrics / "aggregated.jsonl").unlink()
    write_lines(metrics / "rank-000.jsonl", [first])
    state = RunState(discover([run])[0])
    state.refresh()
    if partial_aggregate:
        assert state.series.latest_value("loss/total") == pytest.approx(5.0)

    write_lines(metrics / "rank-001.jsonl", [second])
    state.refresh()
    expected_objective = 7 * 10 * 8 / 9 + 13 * 10 * 32 / 33
    expected = {"total": 20.0, "vision": 7.0, "action": 13.0, "objective": expected_objective}
    record = state.series.last()
    assert record["complete"] is True
    assert record["loss"] == pytest.approx(expected)
    assert record["data"]["samples"] == 4
    assert state.series.latest_value("loss/vision") == pytest.approx(7.0)
    assert state.series.latest_value("loss/objective") == pytest.approx(expected_objective)
    assert state.snapshot()["last"]["loss"] == pytest.approx(20.0)
    assert state.step_detail(1)["loss"] == pytest.approx(expected)
    assert json.loads((metrics / "rank-000.jsonl").read_text())["loss"] == first["loss"]
