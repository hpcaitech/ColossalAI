"""HTTP contracts for the control endpoints.  Skipped where FastAPI is not installed."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from examples.worldmodel.cosmos3_monitor.server import create_app  # noqa: E402
from examples.worldmodel.cosmos3_monitor.store import RunStore  # noqa: E402


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def make_run(tmp_path: Path, *, stopped: bool = False, status: bool = True):
    run = tmp_path / "run"
    metrics = run / "metrics"
    started = time.time() - 100
    write_json(
        metrics / "run.json",
        {
            "run_id": "api-run",
            "world_size": 1,
            "started_at": started,
            "checkpoint_schedule": {"resumable": {"root": str(run / "checkpoints")}},
        },
    )
    write_json(metrics / "heartbeat-000.json", {"rank": 0, "phase": "step_complete", "step": 3, "time": time.time()})
    generation = run / "checkpoints" / "checkpoint-abcd"
    write_json(generation / "complete.json", {"step": 3, "files": {"a": "b"}, "created_at": time.time()})
    write_json(run / "checkpoints" / "latest.json", {"generation": "checkpoint-abcd"})
    if status:
        write_json(
            run / "control" / "status.json",
            {
                "pid": 1,
                "phase": "step_complete",
                "step": 3,
                "run_started_at": started,
                "updated_at": time.time(),
                "request": None,
            },
        )
    if stopped:
        write_json(
            metrics / "terminal-000.json",
            {"rank": 0, "step": 3, "completed": False, "stopped": True, "run_started_at": started, "time": time.time()},
        )
    return run


def client_for(tmp_path: Path, run: Path, **kwargs) -> TestClient:
    store = RunStore([run], state_dir=tmp_path / "state", **kwargs)
    return TestClient(create_app(store))


def test_control_endpoints_round_trip(tmp_path):
    run = make_run(tmp_path)
    with client_for(tmp_path, run) as client:
        token = client.get("/api/runs").json()["data"]["runs"][0]["token"]

        view = client.get(f"/api/runs/{token}/control").json()["data"]
        assert view["can"]["stop"] is True
        assert view["checkpoint"]["generation"] == "checkpoint-abcd"

        response = client.post(f"/api/runs/{token}/control", json={"action": "stop"})
        assert response.status_code == 200
        assert response.json()["data"]["request"]["action"] == "stop"
        assert (run / "control" / "request.json").is_file()

        response = client.delete(f"/api/runs/{token}/control")
        assert response.status_code == 200
        assert response.json()["data"]["request"] is None
        assert not (run / "control" / "request.json").exists()


def test_an_inapplicable_action_is_a_conflict(tmp_path):
    run = make_run(tmp_path, stopped=True)
    with client_for(tmp_path, run) as client:
        token = client.get("/api/runs").json()["data"]["runs"][0]["token"]
        response = client.post(f"/api/runs/{token}/control", json={"action": "stop"})
        assert response.status_code == 409
        assert "not running" in response.json()["detail"]
        response = client.post(f"/api/runs/{token}/control", json={"action": "resume"})
        assert response.status_code == 409
        assert "launch-template" in response.json()["detail"]
        response = client.post(f"/api/runs/{token}/control", json={"action": "explode"})
        assert response.status_code == 409


def test_resume_posts_through_the_launch_template(tmp_path, monkeypatch):
    run = make_run(tmp_path, stopped=True)
    monkeypatch.setattr(
        "examples.worldmodel.cosmos3_monitor.control.subprocess.Popen",
        lambda command, **kwargs: type("P", (), {"pid": 4242, "poll": staticmethod(lambda: None)})(),
    )
    with client_for(tmp_path, run, launch_template="train.sh --resume {checkpoint}") as client:
        token = client.get("/api/runs").json()["data"]["runs"][0]["token"]
        response = client.post(f"/api/runs/{token}/control", json={"action": "resume"})
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["launch"]["state"] == "running"
        assert data["launch"]["command"] == f"train.sh --resume {run / 'checkpoints'}"
        assert json.loads((run / "control" / "request.json").read_text())["action"] == "resume"


def test_unknown_run_is_404(tmp_path):
    run = make_run(tmp_path)
    with client_for(tmp_path, run) as client:
        assert client.get("/api/runs/nope/control").status_code == 404
        assert client.post("/api/runs/nope/control", json={"action": "stop"}).status_code == 404
        assert client.delete("/api/runs/nope/control").status_code == 404
