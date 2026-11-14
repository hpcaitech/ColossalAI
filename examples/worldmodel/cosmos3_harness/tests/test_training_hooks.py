"""Tests for the monitor hooks that live on the training side: heartbeat, ledger and run identity."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from examples.worldmodel.cosmos3_harness.checkpoint_ledger import CheckpointLedger, measure_checkpoint
from examples.worldmodel.cosmos3_harness.run_context import enrich_manifest, sanitize_argv
from examples.worldmodel.cosmos3_harness.training_heartbeat import TrainingHeartbeat, write_terminal
from examples.worldmodel.cosmos3_harness.training_metrics import TrainingMetrics


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_heartbeat_is_atomic_and_stops(tmp_path):
    heartbeat = TrainingHeartbeat(tmp_path, 0, None, interval=0.05, run_id="r1", world_size=4).start()
    heartbeat.update("forward", step=3, microstep=2, microsteps=8)
    deadline = time.time() + 3
    path = tmp_path / "heartbeat-000.json"
    while time.time() < deadline and not path.is_file():
        time.sleep(0.02)
    heartbeat.update("step_complete", step=4)
    heartbeat.close()
    payload = json.loads(path.read_text())
    assert payload["rank"] == 0
    assert payload["owner_pid"] == payload["pid"]
    assert payload["run_id"] == "r1"
    assert payload["world_size"] == 4
    assert payload["phase"] in ("forward", "step_complete")
    assert "memory" in payload
    assert not list(tmp_path.glob("*.tmp"))


def test_terminal_file_is_written(tmp_path):
    write_terminal(tmp_path, 3, {"step": 12, "completed": False})
    payload = json.loads((tmp_path / "terminal-003.json").read_text())
    assert payload["rank"] == 3 and payload["completed"] is False and payload["schema"] == 1


def test_checkpoint_ledger_merges_begin_and_end_and_measures(tmp_path):
    export = tmp_path / "step-10"
    export.mkdir()
    (export / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 4096}, "weight_map": {"a": "x", "b": "y"}})
    )
    ledger = CheckpointLedger(tmp_path, run_id="r1", world_size=8)
    checkpoint_id = ledger.begin(step=10, kind="model_export", path=export, async_=False, schedule_every=10)
    assert ledger.pending == 1
    finished = ledger.end(checkpoint_id, status="complete")
    ledger.close()
    assert finished["bytes"] == 4096 * 2**20  # the index stores MiB, the ledger reports bytes
    assert finished["files"] == 2
    assert finished["source"] == "index"
    assert finished["status"] == "complete"
    rows = read_jsonl(tmp_path / "checkpoints.jsonl")
    assert [row["event"] for row in rows] == ["begin", "end"]
    assert rows[1]["checkpoint_id"] == rows[0]["checkpoint_id"]


def test_measure_checkpoint_prefers_complete_json_total(tmp_path):
    generation = tmp_path / "checkpoint-abc"
    generation.mkdir()
    (generation / "complete.json").write_text(json.dumps({"files": {"a": "sha"}, "total_bytes": 123}))
    assert measure_checkpoint(generation) == (123, 1, "complete.json")


def test_sanitize_argv_redacts_secret_values():
    argv = ["train.py", "--api-key", "secret", "--token=abc", "--data", "/data", "--password"]
    assert sanitize_argv(argv) == [
        "train.py",
        "--api-key",
        "<redacted>",
        "--token=<redacted>",
        "--data",
        "/data",
        "--password",
    ]


def test_enrich_manifest_adds_identity_without_overwriting(tmp_path):
    manifest = enrich_manifest(
        {"run_id": "fixed", "args": {"max_iter": 10}},
        checkpoint_schedule={"model_export": {"every": 100}},
        log_path="/tmp/run.log",
        metrics_every=10,
        log_every=50,
    )
    assert manifest["run_id"] == "fixed"
    assert manifest["run"]["argv"]
    assert manifest["launcher"]["rank"] is not None or manifest["launcher"]["rank"] is None
    assert manifest["checkpoint_schedule"]["model_export"]["every"] == 100
    assert manifest["monitor"]["metrics_every"] == 10
    assert manifest["monitor"]["log_path"] == "/tmp/run.log"


def test_aggregated_manifest_and_sessions_are_persisted(tmp_path):
    metrics = TrainingMetrics(out_dir=tmp_path, rank=0, world_size=1, log_every=1)
    metrics.start({"run_id": "r1", "args": {"max_iter": 2}, "run_name": "unit"})
    metrics.step(1, data={"samples": 8}, loss={"total": 2.0}, perf={"step_ms": 1000.0})
    metrics.finish({"steps": 1, "completed": True})
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["run_id"] == "r1"
    aggregated = read_jsonl(tmp_path / "aggregated.jsonl")
    records = [row for row in aggregated if row.get("type") == "record"]
    assert len(records) == 1
    assert records[0]["expected_world"] == 1
    assert records[0]["complete"] is True
    sessions = read_jsonl(tmp_path / "sessions.jsonl")
    assert {row["type"] for row in sessions} == {"session", "session_end"}
    assert json.loads((tmp_path / "summary.json").read_text())["completed"] is True


def test_aggregate_rejects_duplicate_ranks():
    from examples.worldmodel.cosmos3_harness.training_metrics import aggregate_records, make_record

    with pytest.raises(ValueError, match="duplicate"):
        aggregate_records([make_record(1, 0), make_record(1, 0)])
