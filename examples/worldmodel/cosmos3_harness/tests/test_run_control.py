"""Tests for the file-based stop channel the dashboard drives."""

from __future__ import annotations

import json
import time

import pytest

from examples.worldmodel.cosmos3_harness.run_control import RunControl, read_request


def write_request(run_dir, action="stop", *, request_id="req-1", run_started_at=100.0, requested_at=None):
    payload = {
        "action": action,
        "request_id": request_id,
        "requested_at": time.time() if requested_at is None else requested_at,
        "run_started_at": run_started_at,
    }
    directory = run_dir / "control"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "request.json").write_text(json.dumps(payload))
    return payload


def wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_no_request_means_no_stop(tmp_path):
    control = RunControl(tmp_path, rank=0, world_size=1, poll_seconds=0.05, run_started_at=100.0).start()
    try:
        assert control.boundary(1) is False
        assert control.boundary(2) is False
    finally:
        control.close()


def test_a_matching_request_stops_at_the_next_boundary(tmp_path):
    control = RunControl(tmp_path, rank=0, world_size=1, poll_seconds=0.02, run_started_at=100.0).start()
    try:
        write_request(tmp_path, run_started_at=100.0)
        assert wait_for(lambda: control.stop_pending)
        control.mark("step_complete", step=7)
        assert control.boundary(7) is True
        status = json.loads((tmp_path / "control" / "status.json").read_text())
        assert status["phase"] == "stopping"
        assert status["stop_at_step"] == 7
        assert status["request"]["request_id"] == "req-1"
        # Sticky: the next call does not need the file or the collective.
        assert control.boundary(8) is True
    finally:
        control.close()


def test_a_request_for_another_attempt_is_ignored(tmp_path):
    # The dashboard stamps the run it was looking at; a stop meant for the previous attempt in this
    # directory must not stop the resumed one.
    control = RunControl(tmp_path, rank=0, world_size=1, poll_seconds=0.02, run_started_at=200.0).start()
    try:
        write_request(tmp_path, run_started_at=100.0)
        time.sleep(0.15)
        assert control.stop_pending is False
        assert control.boundary(1) is False
    finally:
        control.close()


def test_a_legacy_request_without_a_stamp_is_honoured(tmp_path):
    control = RunControl(tmp_path, rank=0, world_size=1, poll_seconds=0.02, run_started_at=200.0).start()
    try:
        write_request(tmp_path, run_started_at=None)
        assert wait_for(lambda: control.stop_pending)
        assert control.boundary(1) is True
    finally:
        control.close()


def test_a_malformed_request_is_not_an_error(tmp_path):
    directory = tmp_path / "control"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "request.json").write_text("{not json")
    control = RunControl(tmp_path, rank=0, world_size=1, poll_seconds=0.02, run_started_at=100.0).start()
    try:
        time.sleep(0.1)
        assert control.stop_pending is False
        assert read_request(directory / "request.json") is None
    finally:
        control.close()


def test_resume_requests_are_not_treated_as_stop(tmp_path):
    control = RunControl(tmp_path, rank=0, world_size=1, poll_seconds=0.02, run_started_at=100.0).start()
    try:
        write_request(tmp_path, action="resume")
        time.sleep(0.15)
        assert control.stop_pending is False
        assert control.boundary(1) is False
    finally:
        control.close()


def test_rank_zero_publishes_the_status_and_other_ranks_do_not(tmp_path):
    leader = RunControl(tmp_path, rank=0, world_size=2, poll_seconds=0.02, run_started_at=100.0).start()
    follower = RunControl(tmp_path, rank=1, world_size=2, poll_seconds=0.02, run_started_at=100.0).start()
    try:
        leader.mark("forward", step=5)
        assert wait_for(lambda: (tmp_path / "control" / "status.json").is_file())
        assert wait_for(lambda: json.loads((tmp_path / "control" / "status.json").read_text())["step"] == 5)
        status = json.loads((tmp_path / "control" / "status.json").read_text())
        assert status["run_started_at"] == 100.0
        assert status["phase"] == "forward"
    finally:
        leader.close()
        follower.close()


class _FakeDistributed:
    """Stands in for torch.distributed: the all-reduce reports what the *other* rank saw."""

    ReduceOp = type("ReduceOp", (), {"MAX": "max"})

    def __init__(self, other_rank_saw: bool):
        self.other_rank_saw = other_rank_saw
        self.calls = 0

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def all_reduce(self, tensor, op=None):
        self.calls += 1
        if self.other_rank_saw:
            tensor.fill_(1)


def test_the_agreement_is_collective_not_local(tmp_path, monkeypatch):
    # This rank never saw the request; the all-reduce says its peer did.  Both must stop on the same step.
    fake = _FakeDistributed(other_rank_saw=True)
    monkeypatch.setattr("torch.distributed.is_available", fake.is_available)
    monkeypatch.setattr("torch.distributed.is_initialized", fake.is_initialized)
    monkeypatch.setattr("torch.distributed.all_reduce", fake.all_reduce)
    monkeypatch.setattr("torch.distributed.ReduceOp", fake.ReduceOp, raising=False)
    control = RunControl(tmp_path, rank=1, world_size=2, poll_seconds=5.0, run_started_at=100.0).start()
    try:
        assert control.stop_pending is False
        assert control.boundary(3) is True
        assert fake.calls == 1
    finally:
        control.close()


def test_the_agreement_does_not_run_when_disabled(tmp_path, monkeypatch):
    fake = _FakeDistributed(other_rank_saw=True)
    monkeypatch.setattr("torch.distributed.is_available", fake.is_available)
    monkeypatch.setattr("torch.distributed.is_initialized", fake.is_initialized)
    monkeypatch.setattr("torch.distributed.all_reduce", fake.all_reduce)
    monkeypatch.setattr("torch.distributed.ReduceOp", fake.ReduceOp, raising=False)
    control = RunControl(tmp_path, rank=0, world_size=2, enabled=False, run_started_at=100.0).start()
    try:
        assert control.boundary(1) is False
        assert fake.calls == 0
        assert not (tmp_path / "control").exists()
    finally:
        control.close()


def test_the_step_gate_is_deterministic(tmp_path, monkeypatch):
    fake = _FakeDistributed(other_rank_saw=False)
    monkeypatch.setattr("torch.distributed.is_available", fake.is_available)
    monkeypatch.setattr("torch.distributed.is_initialized", fake.is_initialized)
    monkeypatch.setattr("torch.distributed.all_reduce", fake.all_reduce)
    monkeypatch.setattr("torch.distributed.ReduceOp", fake.ReduceOp, raising=False)
    control = RunControl(tmp_path, rank=0, world_size=2, poll_seconds=5.0, every=5, run_started_at=100.0).start()
    try:
        for step in range(1, 11):
            control.boundary(step)
        # Only steps 5 and 10 ran the collective, and the gate depends on the step alone.
        assert fake.calls == 2
    finally:
        control.close()


def test_close_publishes_the_final_phase(tmp_path):
    control = RunControl(tmp_path, rank=0, world_size=1, poll_seconds=0.02, run_started_at=100.0).start()
    control.mark("stopped", step=42)
    control.close()
    status = json.loads((tmp_path / "control" / "status.json").read_text())
    assert status["phase"] == "stopped"
    assert status["step"] == 42
    assert status["pid"] > 0


def test_an_unwritable_run_directory_disables_controls_without_failing(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise PermissionError("read-only")

    control = RunControl(tmp_path, rank=0, world_size=1, run_started_at=100.0)
    monkeypatch.setattr("pathlib.Path.mkdir", refuse)
    control.start()
    assert control.enabled is False
    assert "PermissionError" in (control.error or "")
    assert control.boundary(1) is False
