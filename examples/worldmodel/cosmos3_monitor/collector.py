"""Per-node sidecar that appends host/NPU telemetry to ``<metrics>/system/node-<host>.jsonl``.

This process is deliberately separate from training.  It samples at a fixed interval, lowers its own
scheduling priority, and degrades to an error record when ``npu-smi`` or ``/proc`` is unavailable.  Nothing
here imports torch or touches the training process.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .system import HostSampler

MAX_FILE_BYTES = 64 * 1024 * 1024


def _rotated(path: Path) -> None:
    try:
        if path.stat().st_size < MAX_FILE_BYTES:
            return
        backup = path.with_suffix(path.suffix + ".1")
        try:
            backup.unlink()
        except OSError:
            pass
        path.rename(backup)
    except OSError:
        pass


def collect_once(
    metrics_dir,
    *,
    node_rank: int = 0,
    disk_paths: Optional[list] = None,
    entry: str = "train_sft.py",
    run_out: str = "",
    sampler: Optional[HostSampler] = None,
) -> Dict[str, Any]:
    Path(metrics_dir)
    sampler = sampler or HostSampler(disk_paths=disk_paths, entry=entry, run_out=run_out)
    payload = sampler.sample(node_rank=node_rank)
    payload["collector"]["interval_s"] = None
    return payload


def run_collector(
    run_dir,
    *,
    metrics_dir: Optional[str] = None,
    interval: float = 5.0,
    node_rank: int = 0,
    disk_paths: Optional[list] = None,
    entry: str = "train_sft.py",
    nice: int = 10,
    max_samples: int = 0,
    stop_event: Optional[Any] = None,
) -> None:
    run_path = Path(run_dir).expanduser()
    metrics = Path(metrics_dir).expanduser() if metrics_dir else run_path / "metrics"
    directory = metrics / "system"
    directory.mkdir(parents=True, exist_ok=True)
    hostname = socket.gethostname()
    path = directory / f"node-{hostname}.jsonl"
    try:
        if nice:
            os.nice(nice)
    except (OSError, AttributeError):
        pass

    stop = stop_event or _install_signals()
    sampler = HostSampler(disk_paths=disk_paths, entry=entry, run_out=str(run_path))
    samples = 0
    while not stop.is_set():
        started = time.time()
        try:
            _rotated(path)
            payload = sampler.sample(node_rank=node_rank)
            payload["collector"]["interval_s"] = interval
            with path.open("a", encoding="utf-8", buffering=1) as stream:
                stream.write(json.dumps(payload, allow_nan=False) + "\n")
        except Exception as exc:  # a collector failure must leave a visible trace, not a silent gap
            try:
                error = {
                    "type": "system",
                    "schema": 1,
                    "time": time.time(),
                    "hostname": hostname,
                    "node_rank": node_rank,
                    "devices": [],
                    "host": {},
                    "collector": {
                        "pid": os.getpid(),
                        "interval_s": interval,
                        "errors": [f"{type(exc).__name__}: {exc}"],
                    },
                    "advisory": {"aicore_utilization_excluded": True},
                }
                with path.open("a", encoding="utf-8", buffering=1) as stream:
                    stream.write(json.dumps(error, allow_nan=False) + "\n")
            except OSError:
                pass
        samples += 1
        if max_samples and samples >= max_samples:
            break
        stop.wait(max(0.2, interval - (time.time() - started)))


class _Stop:
    def __init__(self) -> None:
        self._flag = False

    def set(self) -> None:
        self._flag = True

    def is_set(self) -> bool:
        return self._flag

    def wait(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while not self._flag and time.time() < deadline:
            time.sleep(min(0.2, max(0.01, deadline - time.time())))
        return self._flag


def _install_signals() -> _Stop:
    stop = _Stop()

    def handler(_signum, _frame):
        stop.set()

    try:
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
    except ValueError:
        # Not in the main thread (a test or an embedding process); the caller owns the stop event.
        pass
    return stop
