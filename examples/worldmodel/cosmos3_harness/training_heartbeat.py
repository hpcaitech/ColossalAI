"""A training liveness file independent of the metric log.

The training loop can be inside a long collective or a native checkpoint call where no Python line runs.  A
background thread still publishes a small heartbeat every couple of seconds, so a monitor can tell "the job is
alive but this phase is slow" apart from "the job is gone".  The thread only reads host-side counters; it never
synchronizes a device stream and never writes from a forked DataLoader worker.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["atomic_json", "TrainingHeartbeat", "write_terminal"]


def psutil_cpu_percent() -> Optional[float]:
    try:
        import psutil

        return psutil.cpu_percent()
    except Exception:
        return None


def psutil_available_gib() -> Optional[float]:
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:
        return None


def atomic_json(path, payload: Dict[str, Any]) -> None:
    """Replace ``path`` atomically; a reader never sees a half-written heartbeat."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, default=str, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class TrainingHeartbeat:
    """Publish ``phase``/``step``/memory/process/host at a fixed interval.

    Args:
        directory: Metrics directory; the file is ``heartbeat-<rank:03d>.json``.
        rank: Global rank.
        device: This rank's ``torch.device``; the caller must already have selected it on this thread's process.
        interval: Seconds between samples.  Two seconds is enough to distinguish a slow step from a dead one
            without making the file itself a source of load.
        run_id: Optional run identifier copied into every heartbeat.
    """

    def __init__(
        self,
        directory,
        rank: int,
        device=None,
        interval: float = 2.0,
        *,
        run_id: Optional[str] = None,
        world_size: Optional[int] = None,
        local_rank: Optional[int] = None,
        device_name: Optional[str] = None,
    ) -> None:
        self.directory = Path(directory)
        self.path = self.directory / f"heartbeat-{int(rank):03d}.json"
        self.rank = int(rank)
        self.device = device
        self.interval = max(0.25, float(interval))
        self.run_id = run_id
        self.world_size = world_size
        self.local_rank = local_rank
        self.device_name = device_name
        self.owner_pid = os.getpid()
        now = time.time()
        self.state: Dict[str, Any] = {"phase": "initializing", "phase_time": now, "step": 0, "progress_time": now}
        self.peak = 0.0
        self.error: Optional[str] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{self.rank}", daemon=True)

    def start(self) -> "TrainingHeartbeat":
        self._thread.start()
        return self

    def update(self, phase: str, step: Optional[int] = None, **values: Any) -> None:
        """Record a phase transition and/or a completed step.  This is a memory update only."""
        with self._lock:
            if phase != self.state["phase"]:
                self.state["phase_time"] = time.time()
            if step is not None:
                self.state["step"] = int(step)
                self.state["progress_time"] = time.time()
            self.state["phase"] = phase
            for key, value in values.items():
                if value is not None:
                    self.state[key] = value

    def _memory(self) -> Dict[str, Any]:
        try:
            from .training_metrics import device_memory_gib

            return device_memory_gib()
        except Exception as exc:  # an accelerator that vanished must not kill liveness reporting
            self.error = f"memory: {type(exc).__name__}: {exc}"
            return {}

    def _payload(self, process=None) -> Dict[str, Any]:
        memory = self._memory()
        self.peak = max(self.peak, memory.get("device_used_gib", 0.0) or 0.0)
        if memory:
            memory["sampled_device_peak_gib"] = self.peak
        with self._lock:
            payload: Dict[str, Any] = dict(self.state)
        payload.update(
            schema=1,
            rank=self.rank,
            pid=self.owner_pid,
            owner_pid=self.owner_pid,
            hostname=socket.gethostname(),
            time=time.time(),
            memory=memory,
            memory_sample_interval_s=self.interval,
            error=self.error,
        )
        if self.run_id is not None:
            payload["run_id"] = self.run_id
        if self.world_size is not None:
            payload["world_size"] = int(self.world_size)
        if self.local_rank is not None:
            payload["local_rank"] = int(self.local_rank)
        if self.device is not None:
            index = getattr(self.device, "index", None)
            if index is None:
                index = self.local_rank
            payload["device"] = {"index": index, "name": self.device_name or str(self.device)}
            payload["device_index"] = index
        if process is not None:
            try:
                payload["process"] = {
                    "rss_gib": process.memory_info().rss / 2**30,
                    "cpu_percent": process.cpu_percent(),
                    "threads": process.num_threads(),
                    "read_bytes": process.io_counters().read_bytes,
                    "write_bytes": process.io_counters().write_bytes,
                }
                payload["host"] = {
                    "cpu_percent": psutil_cpu_percent(),
                    "available_gib": psutil_available_gib(),
                }
            except Exception as exc:
                payload["process_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    def _write(self, process=None) -> None:
        if os.getpid() != self.owner_pid:
            return
        atomic_json(self.path, self._payload(process))

    def _run(self) -> None:
        process = None
        try:
            import psutil

            process = psutil.Process()
        except Exception:
            process = None
        # The current accelerator device is thread-local.  Selecting it here makes the memory counters read the
        # right card; it does not change the training thread's selection.
        try:
            import torch

            module = getattr(torch, getattr(self.device, "type", ""), None) if self.device is not None else None
            if module is not None and hasattr(module, "set_device"):
                module.set_device(self.device)
        except Exception as exc:
            self.error = f"device selection: {type(exc).__name__}: {exc}"
        while not self._stop.is_set():
            try:
                if os.getpid() != self.owner_pid:
                    break
                self._write(process)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.interval)

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval + 5)
        # One final synchronous write makes the last phase visible even when the process is about to exit.
        try:
            self._write(None)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"


def write_terminal(directory, rank: int, payload: Dict[str, Any]) -> None:
    """Write this rank's final status; a monitor treats a missing file as "still running or lost"."""
    body = dict(payload)
    body.setdefault("schema", 1)
    body["rank"] = int(rank)
    body.setdefault("time", time.time())
    atomic_json(Path(directory) / f"terminal-{int(rank):03d}.json", body)
