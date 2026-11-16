"""The write side of the dashboard: a file-based control channel into a training run.

Nothing here imports the training package.  The channel is two JSON files inside the run::

    <run>/control/request.json   this module writes it, atomically
    <run>/control/status.json    rank 0 of the training process writes it (cosmos3_harness.run_control)

A ``stop`` request is honoured by the training loop after the current step, which then writes one more
generation and exits ``stopped`` instead of ``failed``.  A ``resume`` request is executed here, by the
server, when it was configured with ``--launch-template``: the template is the command that starts the
run, with ``{checkpoint}`` and ``{run_dir}`` substituted.  A server without a template still records the
request and shows the exact command a person can run instead; it never guesses one.

The request carries ``run_started_at``, copied out of the run manifest.  A training process only honours
a request whose value matches its own, so a stop meant for the previous attempt in the same directory
cannot stop the resumed run -- and no clock is ever compared across machines.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "ControlError",
    "LaunchManager",
    "checkpoint_root",
    "clear_request",
    "control_directory",
    "format_command",
    "latest_generation",
    "read_status",
    "write_request",
]

REQUEST_NAME = "request.json"
STATUS_NAME = "status.json"
#: A status file older than this no longer proves the loop is listening (its writer runs every second).
STATUS_FRESH_SECONDS = 60.0


class ControlError(RuntimeError):
    """The requested action is not applicable now; the message is shown to the user."""

    def __init__(self, message: str, code: str = "rejected") -> None:
        super().__init__(message)
        self.code = code


def control_directory(run_dir) -> Path:
    return Path(run_dir) / "control"


def _read_json(path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def read_status(run_dir) -> Optional[Dict[str, Any]]:
    """The loop's published state, or ``None`` when this run has no control channel."""
    return _read_json(control_directory(run_dir) / STATUS_NAME)


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_request(
    run_dir,
    action: str,
    *,
    run_started_at: Optional[float] = None,
    checkpoint: Optional[str] = None,
    requested_by: str = "monitor",
) -> Dict[str, Any]:
    """Publish a request.  Returns the payload that was written."""
    payload = {
        "schema": 1,
        "action": str(action),
        "request_id": uuid.uuid4().hex,
        "requested_at": time.time(),
        "requested_by": str(requested_by),
        "run_started_at": run_started_at,
    }
    if checkpoint is not None:
        payload["checkpoint"] = str(checkpoint)
    _atomic_write(control_directory(run_dir) / REQUEST_NAME, payload)
    return payload


def clear_request(run_dir) -> bool:
    """Withdraw a request the loop has not seen yet.  Returns whether a file was removed."""
    path = control_directory(run_dir) / REQUEST_NAME
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ControlError(f"cannot withdraw the request: {exc}", code="read_only") from exc


def checkpoint_root(manifest: Dict[str, Any], run_dir) -> Optional[Path]:
    """Where this run's generations live, preferring the path its manifest recorded.

    On a mirror the recorded path points at the training host, so a missing directory falls back to the
    conventional ``<run>/checkpoints`` and then to the run directory itself (the harness's ``--save-dir``).
    """
    schedule = manifest.get("checkpoint_schedule") if isinstance(manifest, dict) else None
    recorded = None
    if isinstance(schedule, dict):
        resumable = schedule.get("resumable")
        if isinstance(resumable, dict):
            recorded = resumable.get("root")
    run_dir = Path(run_dir)
    if recorded:
        candidate = Path(str(recorded))
        if candidate.is_dir():
            return candidate
        if Path(str(recorded)).name and (run_dir / "checkpoints").is_dir():
            return run_dir / "checkpoints"
    for candidate in (run_dir / "checkpoints", run_dir):
        if (candidate / "latest.json").is_file() or (candidate / "complete.json").is_file():
            return candidate
    return None


def latest_generation(root) -> Optional[Dict[str, Any]]:
    """The newest published generation under ``root``, from ``latest.json`` or the newest complete one."""
    if root is None:
        return None
    root = Path(root)
    pointer = _read_json(root / "latest.json")
    names = []
    if pointer and isinstance(pointer.get("generation"), str):
        names.append(pointer["generation"])
    if not names:
        try:
            names = sorted(
                (entry.name for entry in root.iterdir() if entry.name.startswith("checkpoint-") and entry.is_dir()),
                reverse=True,
            )
        except OSError:
            return None
    for name in names:
        if Path(name).name != name:
            continue
        path = root / name
        complete = _read_json(path / "complete.json")
        if complete is None:
            continue
        return {
            "generation": name,
            "path": str(path),
            "root": str(root),
            "step": complete.get("step"),
            "created_at": complete.get("created_at"),
            "bytes": complete.get("total_bytes"),
            "files": len(complete.get("files") or {}),
        }
    return None


def format_command(template: str, *, run_dir, checkpoint) -> str:
    """Render the launch template.  Values are quoted; the template itself is server configuration."""
    return (
        str(template)
        .replace("{checkpoint}", shlex.quote(str(checkpoint)))
        .replace("{run_dir}", shlex.quote(str(run_dir)))
    )


@dataclass
class Launch:
    token: str
    pid: int
    command: str
    started_at: float
    log_path: str
    checkpoint: str
    process: Any = None
    exit_code: Optional[int] = None
    ended_at: Optional[float] = None
    error: Optional[str] = None

    def view(self) -> Dict[str, Any]:
        state = "running"
        if self.process is not None:
            code = self.process.poll()
            if code is not None:
                self.exit_code = code
                if self.ended_at is None:
                    self.ended_at = time.time()
                state = "completed" if code == 0 else "failed"
        elif self.exit_code is not None:
            state = "completed" if self.exit_code == 0 else "failed"
        return {
            "state": state,
            "pid": self.pid,
            "command": self.command,
            "checkpoint": self.checkpoint,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exit_code": self.exit_code,
            "log_path": self.log_path,
            "error": self.error,
        }


class LaunchManager:
    """Start a resumed run from a server-configured command template, and track what it started.

    A launch is deliberately detached (``start_new_session``): the training must outlive the dashboard.
    Its stdout/stderr go to ``<run>/control/launch-<timestamp>.log`` so the Logs tab can tail them.
    """

    def __init__(self, template: Optional[str] = None) -> None:
        self.template = template or None
        self._launches: Dict[str, Launch] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.template)

    def status(self, token: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            launch = self._launches.get(token)
            return launch.view() if launch is not None else None

    def _running(self, launch: Optional[Launch]) -> bool:
        if launch is None:
            return False
        return launch.view()["state"] == "running"

    def spawn(
        self,
        token: str,
        *,
        run_dir,
        checkpoint,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.template:
            raise ControlError("this server has no --launch-template", code="no_launcher")
        with self._lock:
            if self._running(self._launches.get(token)):
                raise ControlError("a resumed run is already starting", code="launch_running")
            command = format_command(self.template, run_dir=run_dir, checkpoint=checkpoint)
            log_path = control_directory(run_dir) / f"launch-{time.strftime('%Y%m%dT%H%M%S')}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                stream = log_path.open("ab")
            except OSError as exc:
                raise ControlError(f"cannot write the launch log: {exc}", code="read_only") from exc
            try:
                process = subprocess.Popen(  # noqa: S602 - the command is server configuration, values are quoted
                    command,
                    shell=True,
                    cwd=str(cwd) if cwd else None,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as exc:
                stream.close()
                raise ControlError(f"cannot start the run: {exc}", code="launch_failed") from exc
            stream.close()
            launch = Launch(
                token=token,
                pid=process.pid,
                command=command,
                started_at=time.time(),
                log_path=str(log_path),
                checkpoint=str(checkpoint),
                process=process,
            )
            self._launches[token] = launch
            return launch.view()
