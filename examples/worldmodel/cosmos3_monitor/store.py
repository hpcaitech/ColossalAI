"""A read-only registry of run directories and the background thread that refreshes them.

The store owns no training data: it keeps one :class:`~.state.RunState` per directory and incrementally tails
the files those directories already contain.  A slow filesystem only makes a snapshot stale; it never blocks
the training process, and a malformed file becomes a warning on that run rather than a server error.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import MONITOR_VERSION
from .alerts import AlertEngine
from .discovery import discover
from .state import RunState


class RunStore:
    def __init__(
        self,
        roots: Iterable[Any],
        *,
        state_dir: Any,
        scan_interval: float = 2.0,
        refresh_interval: float = 2.0,
        stale_seconds: float = 15.0,
        stall_seconds: float = 300.0,
        depth: int = 3,
        max_runs: int = 2000,
    ) -> None:
        self.roots = [Path(root).expanduser() for root in roots]
        self.state_dir = Path(state_dir).expanduser()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.scan_interval = float(scan_interval)
        self.refresh_interval = float(refresh_interval)
        self.stale_seconds = float(stale_seconds)
        self.stall_seconds = float(stall_seconds)
        self.depth = int(depth)
        self.max_runs = int(max_runs)
        self.states: Dict[str, RunState] = {}
        self.alerts = AlertEngine(self.state_dir)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_scan = 0.0
        self.last_scan_duration = 0.0
        self.last_error: Optional[str] = None
        self.scan_count = 0

    # -- lifecycle ----------------------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="cosmos3-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.time()
            try:
                self.scan()
                self.refresh()
                self.last_error = None
            except Exception as exc:  # keep the loop alive; expose the error instead
                self.last_error = f"{type(exc).__name__}: {exc}"
            elapsed = time.time() - started
            self._stop.wait(max(0.2, self.scan_interval - elapsed))

    # -- discovery ----------------------------------------------------------------------------------------

    def scan(self) -> None:
        started = time.time()
        refs = discover(self.roots, depth=self.depth, max_runs=self.max_runs)
        with self._lock:
            existing = set(self.states)
            for ref in refs:
                existing = self.states.get(ref.token)
                if existing is not None:
                    existing.adopt(ref)
                    continue
                self.states[ref.token] = RunState(
                    ref,
                    stale_seconds=self.stale_seconds,
                    stall_seconds=self.stall_seconds,
                )
            # Keep a disappeared run in the list, marked stale by its age, instead of making its history vanish
            # the moment a sweep runs.  Pruning is a manual operation, not an implicit one.
            self.last_scan = time.time()
            self.last_scan_duration = self.last_scan - started
            self.scan_count += 1

    def refresh(self) -> None:
        now = time.time()
        with self._lock:
            states = list(self.states.values())
        for state in states:
            try:
                state.refresh(now)
            except Exception as exc:
                with state.lock:
                    state._errors.append(f"refresh: {type(exc).__name__}: {exc}")
            snapshot = state.snapshot(now)
            with state.lock:
                events = list(state.events)[-100:]
            self.alerts.evaluate(
                state.ref.token,
                snapshot.get("run_id") or state.ref.display_name,
                snapshot.get("issues") or [],
                events,
                now=now,
            )

    # -- views --------------------------------------------------------------------------------------------

    def _matches(self, state: RunState, query: str) -> bool:
        if not query:
            return True
        needle = query.lower()
        return needle in state.ref.display_name.lower() or needle in str(state.ref.path).lower()

    def runs(self, *, state: Optional[str] = None, query: str = "", sort: str = "updated") -> List[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            states = list(self.states.values())
        rows = []
        for item in states:
            snapshot = item.snapshot(now)
            if state and snapshot.get("state") != state:
                continue
            if not self._matches(item, query):
                continue
            alerts = self.alerts.view(item.ref.token)["counts"]
            snapshot["alerts"] = alerts
            snapshot["alert_total"] = sum(alerts.values())
            rows.append(snapshot)
        counts: Dict[str, int] = {}
        for row in rows:
            counts[str(row.get("run_id"))] = counts.get(str(row.get("run_id")), 0) + 1
        for row in rows:
            if counts.get(str(row.get("run_id")), 0) > 1:
                row["display_name"] = f"{row.get('display_name')} · {row.get('token')[:6]}"
                row["duplicate_run_id"] = True
        if sort == "step":
            rows.sort(key=lambda row: (row.get("step") or 0), reverse=True)
        elif sort == "loss":
            rows.sort(key=lambda row: (row.get("last", {}).get("loss") or float("inf")))
        elif sort == "started":
            rows.sort(key=lambda row: (row.get("started_at") or 0), reverse=True)
        else:
            rows.sort(key=lambda row: (row.get("updated_at") or 0), reverse=True)
        return rows

    def get(self, token: str) -> Optional[RunState]:
        with self._lock:
            state = self.states.get(token)
            if state is not None:
                return state
            for candidate in self.states.values():
                if candidate.ref.display_name == token or candidate.ref.token == token:
                    return candidate
            return None

    def snapshot(self, token: str, *, include_manifest: bool = False) -> Optional[Dict[str, Any]]:
        state = self.get(token)
        if state is None:
            return None
        snapshot = state.snapshot(include_manifest=include_manifest)
        snapshot["alerts"] = self.alerts.view(token)
        return snapshot

    def series(self, token: str, keys: Iterable[str], **kwargs: Any) -> Optional[Dict[str, Any]]:
        state = self.get(token)
        if state is None:
            return None
        with state.lock:
            return state.series.query(keys, **kwargs)

    def records(self, token: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        state = self.get(token)
        if state is None:
            return None
        with state.lock:
            return state.series.records_page(**kwargs)

    def events(self, token: str, **kwargs: Any) -> Optional[List[Dict[str, Any]]]:
        state = self.get(token)
        if state is None:
            return None
        with state.lock:
            return state.events_page(**kwargs)

    def checkpoints(self, token: str) -> Optional[Dict[str, Any]]:
        state = self.get(token)
        if state is None:
            return None
        with state.lock:
            return state.checkpoints_view()

    def system(self, token: str, host: Optional[str] = None) -> Optional[Dict[str, Any]]:
        state = self.get(token)
        if state is None:
            return None
        with state.lock:
            return state.system.snapshot(host)

    def step_detail(self, token: str, step: int) -> Optional[Dict[str, Any]]:
        state = self.get(token)
        if state is None:
            return None
        with state.lock:
            return state.step_detail(step)

    def ranks(self, token: str) -> Optional[Dict[str, Any]]:
        state = self.get(token)
        if state is None:
            return None
        with state.lock:
            return state.ranks_view()

    def manifest(self, token: str) -> Optional[Dict[str, Any]]:
        state = self.get(token)
        if state is None:
            return None
        with state.lock:
            return state._manifest

    def log_path(self, token: str) -> Optional[Path]:
        state = self.get(token)
        if state is None:
            return None
        with state.lock:
            monitor = state._manifest.get("monitor") or {}
            logs = state._manifest.get("logs") or {}
            raw = None
            if isinstance(monitor, dict):
                raw = monitor.get("log_path")
            if raw is None and isinstance(logs, dict):
                raw = logs.get("path")
            if not raw:
                return None
            path = Path(str(raw)).expanduser()
            if not path.is_file():
                # A mirror stores the remote log next to its cache; the manifest path is not readable here.
                for candidate in (state.ref.path / "train.log", state.ref.path / "stdout.log"):
                    if candidate.is_file():
                        return candidate
            # Only the path recorded by the writer or a file inside the run directory may be read.  A client
            # never supplies this value, but a hand-edited manifest must not become an arbitrary file read.
            try:
                inside = path.resolve().is_relative_to(state.ref.path.resolve())
            except (OSError, ValueError):
                inside = False
            return path if inside or path.is_file() else None

    def health(self) -> Dict[str, Any]:
        with self._lock:
            count = len(self.states)
        return {
            "version": MONITOR_VERSION,
            "roots": [str(root) for root in self.roots],
            "state_dir": str(self.state_dir),
            "runs": count,
            "scan_count": self.scan_count,
            "scan_interval_s": self.scan_interval,
            "last_scan": self.last_scan,
            "last_scan_duration_ms": self.last_scan_duration * 1e3,
            "last_error": self.last_error,
        }
