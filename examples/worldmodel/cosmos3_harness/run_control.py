"""A file-based stop channel between the training loop and the dashboard.

The dashboard usually runs in another process -- often on another host -- and cannot signal the
training ranks directly, so the channel is a directory inside the run:

    <run>/control/request.json   written by the dashboard: {"action": "stop", ...}
    <run>/control/status.json    written by rank 0: what the loop is doing, which request it saw

A watcher thread reads the request; the training thread only ever reads a boolean, so a slow or
half-mounted filesystem can make a stop late but never makes a step slow.

A stop must be honoured at the same step on every rank.  Leaving the loop at different iterations
desynchronises the next collective -- one rank's gradient all-reduce would pair with another
rank's checkpoint all-gather -- and the job hangs.  :meth:`RunControl.boundary` therefore runs one
four-byte all-reduce per step whose maximum is "some rank has seen a stop request"; every rank
gets the same answer on the same iteration and leaves together.  A run with no dashboard attached
pays that single all-reduce per step; ``--no-graceful-stop`` removes it for a benchmark.

A request written before this process started belongs to the previous attempt.  Clocks on a
workstation and on a training node disagree, so the request carries the ``run_started_at`` value
copied out of the run manifest and the loop only honours requests whose value matches its own --
one number, read from the same file by both sides, never compared across machines.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:  # imported as a module of the example package
    from .training_heartbeat import atomic_json
except ImportError:  # direct execution from the example directory
    from training_heartbeat import atomic_json

__all__ = ["ControlRequest", "RunControl", "control_directory", "read_request"]

#: A request may also be honoured when it carries no run stamp at all (a hand-written file).
_CLOCK_TOLERANCE = 1e-3


def control_directory(run_dir) -> Path:
    """``<run>/control``: the directory the dashboard writes into and the monitor reads."""
    return Path(run_dir) / "control"


@dataclass(frozen=True)
class ControlRequest:
    action: str
    request_id: str = ""
    requested_at: float = 0.0
    checkpoint: Optional[str] = None
    run_started_at: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def read_request(path) -> Optional[ControlRequest]:
    """Parse ``request.json``; an unreadable or malformed file is "no request", never an error."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("action") not in ("stop", "resume"):
        return None
    try:
        requested_at = float(payload.get("requested_at") or 0.0)
    except (TypeError, ValueError):
        requested_at = 0.0
    run_started_at = payload.get("run_started_at")
    try:
        run_started_at = float(run_started_at) if run_started_at is not None else None
    except (TypeError, ValueError):
        run_started_at = None
    return ControlRequest(
        action=str(payload["action"]),
        request_id=str(payload.get("request_id") or ""),
        requested_at=requested_at,
        checkpoint=payload.get("checkpoint"),
        run_started_at=run_started_at,
        raw=payload,
    )


class RunControl:
    """Watch ``<run>/control/request.json`` and agree a stop step across ranks.

    Args:
        run_dir: The run directory (the one that holds ``metrics/``), not the metrics directory.
        rank: Global rank; only rank 0 writes ``status.json``.
        world_size: Number of ranks taking part in the agreement.
        device: This rank's ``torch.device``, used for the agreement tensor.  ``None`` or a CPU
            device keeps it on the host, which is what a Gloo/CPU test wants.
        enabled: ``False`` turns the channel off completely (no thread, no collective).
        poll_seconds: How often the watcher thread re-reads the request file.
        every: Run the agreement every N steps.  The gate is ``step % every`` on the step number,
            which is identical on every rank; gating on anything rank-local would desynchronise
            the collective instead.
        run_started_at: The manifest's ``started_at``; requests stamped with a different value
            come from an earlier attempt in the same directory and are ignored.
        run_id: Copied into ``status.json`` for the dashboard.
    """

    def __init__(
        self,
        run_dir,
        *,
        rank: int = 0,
        world_size: int = 1,
        device=None,
        enabled: bool = True,
        poll_seconds: float = 1.0,
        every: int = 1,
        run_started_at: Optional[float] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.directory = control_directory(run_dir)
        self.request_path = self.directory / "request.json"
        self.status_path = self.directory / "status.json"
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.device = device
        self.enabled = bool(enabled)
        self.poll_seconds = max(0.25, float(poll_seconds))
        self.every = max(1, int(every))
        self.run_started_at = run_started_at
        self.run_id = run_id
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seen: set = set()
        self._request: Optional[ControlRequest] = None
        self._agreed = False
        self._agree_step: Optional[int] = None
        self._phase = "initializing"
        self._step = 0
        self._last_published: Any = None
        self._flag = None
        self.error: Optional[str] = None

    # -- lifecycle ----------------------------------------------------------------------------------------

    def start(self) -> "RunControl":
        if not self.enabled:
            return self
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # An unwritable run directory costs the dashboard its controls, never the training run.
            self.error = f"{type(exc).__name__}: {exc}"
            self.enabled = False
            return self
        self._publish(force=True)
        self._thread = threading.Thread(target=self._run, name=f"run-control-{self.rank}", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.poll_seconds + 5)
        self._publish(force=True)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll()
            except Exception as exc:  # a broken control file must not kill the thread
                self.error = f"{type(exc).__name__}: {exc}"
            self._publish()
            self._stop_event.wait(self.poll_seconds)

    # -- request handling ---------------------------------------------------------------------------------

    def _matches_this_run(self, request: ControlRequest) -> bool:
        if request.run_started_at is None or self.run_started_at is None:
            return True
        return abs(request.run_started_at - self.run_started_at) <= _CLOCK_TOLERANCE

    def _poll(self) -> None:
        request = read_request(self.request_path)
        if request is None or request.request_id in self._seen:
            return
        self._seen.add(request.request_id)
        if request.action != "stop" or not self._matches_this_run(request):
            return
        with self._lock:
            if self._request is None or request.requested_at >= self._request.requested_at:
                self._request = request

    @property
    def request(self) -> Optional[ControlRequest]:
        with self._lock:
            return self._request

    @property
    def stop_pending(self) -> bool:
        """Whether this rank has seen a stop request.  Memory read; no filesystem, no collective."""
        with self._lock:
            return self._request is not None

    @property
    def agreed(self) -> bool:
        return self._agreed

    # -- the collective -----------------------------------------------------------------------------------

    def _agreement_tensor(self):
        if self._flag is None:
            import torch

            device = self.device
            if device is None or getattr(device, "type", "cpu") == "cpu":
                device = torch.device("cpu")
            self._flag = torch.zeros(1, dtype=torch.int32, device=device)
        return self._flag

    def boundary(self, step: int) -> bool:
        """Called once per step by every rank.  ``True`` means every rank stops after this step.

        The call site must be the same program point on every rank -- one call per step, after the
        optimizer step -- so that the *N*-th all-reduce pairs the *N*-th iteration everywhere.
        """
        if not self.enabled:
            return False
        if self._agreed:
            return True
        if self.every > 1 and step % self.every:
            return False
        local = self.stop_pending
        if self.world_size <= 1:
            agreed = local
        else:
            import torch.distributed as dist

            if not dist.is_available() or not dist.is_initialized():
                agreed = local
            else:
                flag = self._agreement_tensor()
                flag.fill_(1 if local else 0)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                agreed = bool(int(flag.item()))
        if agreed:
            with self._lock:
                self._agreed = True
                self._agree_step = int(step)
                self._phase = "stopping"
            self._publish(force=True)
        return self._agreed

    # -- status -------------------------------------------------------------------------------------------

    def mark(self, phase: str, step: Optional[int] = None) -> None:
        """Record the loop's phase.  Memory only; rank 0's watcher thread writes it out."""
        with self._lock:
            self._phase = str(phase)
            if step is not None:
                self._step = int(step)

    def _payload(self) -> Dict[str, Any]:
        with self._lock:
            request = self._request
            payload = {
                "schema": 1,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "run_id": self.run_id,
                "run_started_at": self.run_started_at,
                "phase": self._phase,
                "step": self._step,
                "stop_at_step": self._agree_step,
                "request": (
                    None
                    if request is None
                    else {
                        "action": request.action,
                        "request_id": request.request_id,
                        "requested_at": request.requested_at,
                    }
                ),
                "updated_at": time.time(),
            }
        return payload

    def _publish(self, force: bool = False) -> None:
        if not self.enabled or self.rank != 0:
            return
        payload = self._payload()
        signature = (
            payload["phase"],
            payload["step"],
            payload["stop_at_step"],
            (payload["request"] or {}).get("request_id"),
        )
        if not force and signature == self._last_published:
            return
        try:
            atomic_json(self.status_path, payload)
            self._last_published = signature
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"
