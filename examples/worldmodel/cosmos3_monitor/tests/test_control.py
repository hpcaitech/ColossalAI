"""Tests for the dashboard's control channel: what it offers, what it refuses, and what it writes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from examples.worldmodel.cosmos3_monitor.control import (
    ControlError,
    checkpoint_root,
    clear_request,
    latest_generation,
    read_status,
    write_request,
)
from examples.worldmodel.cosmos3_monitor.store import RunStore


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def make_run(
    tmp_path: Path,
    *,
    started_at: float = None,
    with_checkpoint: bool = True,
    status=None,
    terminal=None,
    world: int = 1,
):
    run = tmp_path / "run"
    metrics = run / "metrics"
    started = time.time() - 100 if started_at is None else started_at
    write_json(
        metrics / "run.json",
        {
            "run_id": "run-ctl",
            "run_name": "control",
            "world_size": world,
            "started_at": started,
            "args": {"max_iter": 10},
            "checkpoint_schedule": {"resumable": {"root": str(run / "checkpoints")}},
        },
    )
    write_json(metrics / "heartbeat-000.json", {"rank": 0, "phase": "step_complete", "step": 3, "time": time.time()})
    if with_checkpoint:
        generation = run / "checkpoints" / "checkpoint-abcd"
        write_json(
            generation / "complete.json",
            {"step": 3, "total_bytes": 1024, "files": {"a": "b"}, "created_at": time.time()},
        )
        write_json(run / "checkpoints" / "latest.json", {"generation": "checkpoint-abcd"})
    if status is not None:
        write_json(run / "control" / "status.json", status)
    if terminal is not None:
        write_json(metrics / "terminal-000.json", terminal)
    return run


def store_for(tmp_path: Path, run: Path, **kwargs) -> RunStore:
    store = RunStore([run], state_dir=tmp_path / "state", **kwargs)
    store.scan()
    store.refresh()
    return store


def live_status(started_at: float, **extra):
    payload = {
        "schema": 1,
        "pid": 4242,
        "phase": "step_complete",
        "step": 3,
        "stop_at_step": None,
        "request": None,
        "run_started_at": started_at,
        "updated_at": time.time(),
    }
    payload.update(extra)
    return payload


def test_stop_needs_a_live_control_channel(tmp_path):
    run = make_run(tmp_path)
    store = store_for(tmp_path, run)
    view = store.control_view(store.runs()[0]["token"])
    assert view["can"]["stop"] is False
    assert view["reasons"]["stop"] == "no_channel"
    assert view["checkpoint"]["generation"] == "checkpoint-abcd"
    assert view["checkpoint"]["step"] == 3


def test_stop_request_is_stamped_with_the_run_and_can_be_withdrawn(tmp_path):
    started = time.time() - 100
    run = make_run(tmp_path, started_at=started, status=live_status(started))
    store = store_for(tmp_path, run)
    token = store.runs()[0]["token"]
    assert store.control_view(token)["can"]["stop"] is True

    view = store.request_control(token, "stop")
    request = json.loads((run / "control" / "request.json").read_text())
    assert request["action"] == "stop"
    # The training process compares this value with its own manifest stamp; the clock is never compared.
    assert request["run_started_at"] == pytest.approx(started)
    assert view["request"]["action"] == "stop"
    assert view["can"]["cancel"] is True
    assert view["can"]["stop"] is False  # one outstanding request is enough

    view = store.cancel_control(token)
    assert view["request"] is None
    assert not (run / "control" / "request.json").exists()


def test_acknowledged_stop_is_not_offered_again(tmp_path):
    started = time.time() - 100
    run = make_run(tmp_path, started_at=started, status=live_status(started))
    store = store_for(tmp_path, run)
    token = store.runs()[0]["token"]
    store.request_control(token, "stop")
    request = json.loads((run / "control" / "request.json").read_text())
    write_json(
        run / "control" / "status.json",
        live_status(
            started, phase="stopping", stop_at_step=4, request={"action": "stop", "request_id": request["request_id"]}
        ),
    )
    store.refresh()
    view = store.control_view(token)
    assert view["can"]["stop"] is False
    assert view["can"]["cancel"] is False
    assert view["reasons"]["stop"] == "already_stopping"


def test_resume_is_refused_without_a_launcher_but_reports_the_command(tmp_path):
    started = time.time() - 100
    run = make_run(
        tmp_path,
        started_at=started,
        terminal={
            "rank": 0,
            "step": 3,
            "completed": False,
            "stopped": True,
            "run_started_at": started,
            "time": time.time(),
        },
    )
    store = store_for(tmp_path, run)
    token = store.runs()[0]["token"]
    view = store.control_view(token)
    assert view["run_state"] == "stopped"
    assert view["can"]["resume"] is False
    assert view["reasons"]["resume"] == "no_launcher"
    assert view["manual_command"] == f"--resume {run / 'checkpoints'}"
    # A stopped run has no ETA left and its elapsed time no longer grows with the dashboard's clock.
    snapshot = store.snapshot(token)
    assert snapshot["eta_s"] == 0.0
    assert snapshot["eta_source"] == "stopped"


def test_resume_renders_the_launch_template_and_rejects_foreign_checkpoints(tmp_path):
    started = time.time() - 100
    run = make_run(
        tmp_path,
        started_at=started,
        terminal={
            "rank": 0,
            "step": 3,
            "completed": False,
            "stopped": True,
            "run_started_at": started,
            "time": time.time(),
        },
    )
    store = store_for(tmp_path, run, launch_template="echo resume {checkpoint} in {run_dir}")
    token = store.runs()[0]["token"]
    view = store.control_view(token)
    assert view["can"]["resume"] is True
    assert view["resume_command"] == f"echo resume {run / 'checkpoints'} in {run}"

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(ControlError) as excinfo:
        store.request_control(token, "resume", checkpoint=str(outside))
    assert excinfo.value.code == "bad_checkpoint"

    with pytest.raises(ControlError) as excinfo:
        store.request_control(token, "stop")
    assert excinfo.value.code == "not_running"


def test_resume_spawns_the_template_and_records_the_request(tmp_path, monkeypatch):
    started = time.time() - 100
    run = make_run(
        tmp_path,
        started_at=started,
        terminal={
            "rank": 0,
            "step": 3,
            "completed": False,
            "stopped": True,
            "run_started_at": started,
            "time": time.time(),
        },
    )
    launched = {}

    class FakeProcess:
        pid = 999

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("examples.worldmodel.cosmos3_monitor.control.subprocess.Popen", fake_popen)
    store = store_for(tmp_path, run, launch_template="train.sh --resume {checkpoint}")
    token = store.runs()[0]["token"]
    view = store.request_control(token, "resume")
    assert launched["command"] == f"train.sh --resume {run / 'checkpoints'}"
    assert launched["kwargs"]["start_new_session"] is True
    assert view["launch"]["state"] == "running"
    assert view["launch"]["pid"] == 999
    request = json.loads((run / "control" / "request.json").read_text())
    assert request["action"] == "resume"
    assert request["checkpoint"] == str(run / "checkpoints")


def test_resume_can_name_an_individual_generation(tmp_path, monkeypatch):
    started = time.time() - 100
    run = make_run(
        tmp_path,
        started_at=started,
        terminal={
            "rank": 0,
            "step": 3,
            "completed": False,
            "stopped": True,
            "run_started_at": started,
            "time": time.time(),
        },
    )
    monkeypatch.setattr(
        "examples.worldmodel.cosmos3_monitor.control.subprocess.Popen",
        lambda command, **kwargs: type("P", (), {"pid": 1, "poll": staticmethod(lambda: None)})(),
    )
    store = store_for(tmp_path, run, launch_template="train.sh --resume {checkpoint}")
    token = store.runs()[0]["token"]
    generation = run / "checkpoints" / "checkpoint-abcd"
    view = store.request_control(token, "resume", checkpoint=str(generation))
    assert view["launch"]["checkpoint"] == str(generation)


def test_a_mirrored_run_offers_no_controls(tmp_path):
    # The cache is writable, but the training process reads the real run directory on another host.
    started = time.time() - 100
    run = make_run(tmp_path, started_at=started, status=live_status(started))
    write_json(run / "mirror-state.json", {"nodes": {}})
    store = store_for(tmp_path, run)
    view = store.control_view(store.runs()[0]["token"])
    assert view["can"]["stop"] is False
    assert view["reasons"]["stop"] == "mirrored"
    with pytest.raises(ControlError):
        store.request_control(store.runs()[0]["token"], "stop")


def test_control_is_disabled_on_a_read_only_server(tmp_path):
    started = time.time() - 100
    run = make_run(tmp_path, started_at=started, status=live_status(started))
    store = store_for(tmp_path, run, control_enabled=False)
    view = store.control_view(store.runs()[0]["token"])
    assert view["available"] is False
    assert view["can"]["stop"] is False
    assert view["reasons"]["stop"] == "control_disabled"
    with pytest.raises(ControlError):
        store.request_control(store.runs()[0]["token"], "stop")


def test_a_terminal_from_the_previous_attempt_is_ignored(tmp_path):
    started = time.time() - 10
    run = make_run(
        tmp_path,
        started_at=started,
        terminal={
            "rank": 0,
            "step": 90,
            "completed": False,
            "stopped": True,
            "run_started_at": started - 5000,
            "time": time.time() - 4000,
        },
    )
    store = store_for(tmp_path, run)
    view = store.control_view(store.runs()[0]["token"])
    # The stale terminal belongs to the attempt that was stopped before this one started.
    assert view["run_state"] == "running"


def test_a_summary_from_the_previous_attempt_is_ignored(tmp_path):
    run = make_run(tmp_path, started_at=time.time() - 10)
    metrics = run / "metrics"
    (metrics / "aggregated.jsonl").write_text(
        json.dumps(
            {"type": "summary", "time": time.time() - 4000, "summary": {"completed": True, "run_started_at": 1.0}}
        )
        + "\n"
    )
    store = store_for(tmp_path, run)
    view = store.control_view(store.runs()[0]["token"])
    assert view["run_state"] == "running"


def test_latest_generation_falls_back_when_the_pointer_is_missing(tmp_path):
    run = make_run(tmp_path)
    (run / "checkpoints" / "latest.json").unlink()
    root = checkpoint_root({"checkpoint_schedule": {"resumable": {"root": str(run / "checkpoints")}}}, run)
    generation = latest_generation(root)
    assert generation["generation"] == "checkpoint-abcd"
    assert generation["files"] == 1


def test_request_files_round_trip(tmp_path):
    run = make_run(tmp_path)
    payload = write_request(run, "stop", run_started_at=12.5)
    assert read_status(run) is None
    assert json.loads((run / "control" / "request.json").read_text())["request_id"] == payload["request_id"]
    assert clear_request(run) is True
    assert clear_request(run) is False
