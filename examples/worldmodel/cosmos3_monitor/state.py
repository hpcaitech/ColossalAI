"""Per-run incremental state: records, heartbeats, terminals, checkpoints and the health state machine.

The state machine is intentionally close to the one in the validation worktree's ``monitor_training.py``:
a run that reports an error is ``failed``; a run with a stale heartbeat or a stalled rank is ``attention``;
a run whose every terminal file says completed is ``completed``.  The difference is that everything here is
incremental and reads the current worktree's files as well as the validation worktree's.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .discovery import RunRef, derive_topology, load_manifest
from .discovery import max_steps as manifest_max_steps
from .discovery import world_size
from .readers import JsonFile, JsonlTail, file_age, read_json
from .series import SeriesStore
from .system import SystemHistory, read_system_files

STALE_SECONDS = 15.0
STALL_SECONDS = 300.0
CHECKPOINT_PENDING_SECONDS = 900.0


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the nested record the training loop writes into ``section/key`` leaves."""
    flat: Dict[str, Any] = {}

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{prefix}/{key}" if prefix else str(key))
        else:
            flat[prefix] = value

    walk(record, "")
    return flat


class RunState:
    """Everything the dashboard knows about one run directory."""

    def __init__(
        self, ref: RunRef, *, stale_seconds: float = STALE_SECONDS, stall_seconds: float = STALL_SECONDS
    ) -> None:
        self.ref = ref
        self.stale_seconds = float(stale_seconds)
        self.stall_seconds = float(stall_seconds)
        self.manifest = JsonFile(ref.manifest_path) if ref.manifest_path else None
        self.config = JsonFile(ref.config_path) if ref.config_path else None
        self.aggregated = JsonlTail(ref.metrics_dir / "aggregated.jsonl")
        self.checkpoint_tail = JsonlTail(ref.metrics_dir / "checkpoints.jsonl")
        self.rank_tails: Dict[int, JsonlTail] = {}
        self.heartbeat_files: Dict[int, JsonFile] = {}
        self.terminal_files: Dict[int, JsonFile] = {}
        self.launcher_files: Dict[str, JsonFile] = {}
        self.series = SeriesStore()
        self.system = SystemHistory()
        self.events: Deque[Dict[str, Any]] = deque(maxlen=4000)
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.pending_ranks: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self.summary: Optional[Dict[str, Any]] = None
        self._last_record: Optional[Dict[str, Any]] = None
        self._last_refresh = 0.0
        self._manifest: Dict[str, Any] = {}
        self._run_id: Optional[str] = None
        self._world = 0
        self._max_steps: Optional[int] = None
        self._started_at: Optional[float] = None
        self._checkpoint_scan_at = 0.0
        self._checkpoint_dirs: Dict[str, float] = {}
        self._errors: List[str] = []
        self._aggregated_seen = False
        self._fallback_partial_at: Dict[int, float] = {}
        self.lock = threading.RLock()

    def adopt(self, ref: RunRef) -> None:
        """Upgrade a run discovered before its ``run.json`` appeared (e.g. a mirror's first poll).

        The config-only layout and the metrics layout describe the same run; adopting the fuller manifest
        must not look like a new run, so the identity reset is armed only for a genuine manifest change.
        """
        if ref.manifest_path is not None and self.ref.manifest_path != ref.manifest_path:
            self.ref.manifest_path = ref.manifest_path
            self.ref.layout = ref.layout
            self.manifest = JsonFile(ref.manifest_path)
            self._run_id = None
        if ref.config_path is not None and self.ref.config_path != ref.config_path:
            self.ref.config_path = ref.config_path
            self.config = JsonFile(ref.config_path)

    # -- refresh ------------------------------------------------------------------------------------------

    def refresh(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self.lock:
            self._refresh_locked(now)

    def _refresh_locked(self, now: float) -> None:
        self._last_refresh = now
        self._manifest = self._read_manifest()
        run_id = self._manifest.get("run_id")
        if self._run_id is not None and run_id is not None and run_id != self._run_id:
            # The same directory was reused for a new run: keep the path token but drop the old history.
            self.series.clear()
            self.aggregated.reset()
            self.rank_tails.clear()
            self.pending_ranks.clear()
            self.events.clear()
            self.checkpoints.clear()
            self.checkpoint_tail.reset()
            self.summary = None
            self._last_record = None
            self.system = SystemHistory()
        self._run_id = run_id or self._run_id
        self._world = world_size(self._manifest) or self._world
        self._max_steps = manifest_max_steps(self._manifest) or self._max_steps
        self._started_at = _num(self._manifest.get("started_at")) or self._started_at
        self.series.set_world(self._world)
        self._errors = []
        try:
            self._read_aggregated()
            if not self._aggregated_seen:
                self._read_rank_fallback(now)
            self._read_heartbeats()
            self._read_terminals()
            self._read_launchers()
            self._read_checkpoint_ledger(now)
            read_system_files(self.ref.metrics_dir, self.system)
        except Exception as exc:  # a monitor must never take the server down with it
            self._errors.append(f"{type(exc).__name__}: {exc}")

    def _read_manifest(self) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {}
        if self.manifest is not None:
            value = self.manifest.read()
            if isinstance(value, dict):
                manifest.update(value)
        if self.config is not None:
            value = self.config.read()
            if isinstance(value, dict):
                manifest.setdefault("config", value)
                manifest.setdefault("args", _args_to_dict(value.get("args")))
                for key in ("run_id", "nodes", "devices", "out", "repo", "master_addr", "master_port"):
                    manifest.setdefault(key, value.get(key))
        if not manifest:
            # A mirror may only have written config.json under another name; fall back to the ref.
            manifest = load_manifest(self.ref)
        return manifest

    def _read_aggregated(self) -> None:
        path = self.ref.metrics_dir / "aggregated.jsonl"
        if not path.is_file():
            return
        self._aggregated_seen = True
        if self.aggregated.truncated:
            self.series.clear()
            self._last_record = None
            self.pending_ranks.clear()
            for tail in self.rank_tails.values():
                tail.reset()
            self.aggregated.truncated = False
        for record in self.aggregated.read_new():
            kind = record.get("type")
            if kind == "record":
                if self.series.append(record):
                    self._last_record = record
            elif kind == "event":
                self._add_event(record)
            elif kind == "summary":
                self.summary = record.get("summary") or record

    def _read_rank_fallback(self, now: float) -> None:
        """Aggregate per-rank files when a run predates ``aggregated.jsonl``.

        A step is published when every expected rank has written it, or, if the run is old enough, when a
        later step has appeared.  The latter is a partial view and is marked ``complete: false``, exactly as
        the training-side aggregator does.
        """
        paths = sorted(self.ref.metrics_dir.glob("rank-*.jsonl"))
        if not paths:
            return
        for path in paths:
            try:
                rank = int(path.stem.split("-")[-1])
            except (ValueError, IndexError):
                continue
            tail = self.rank_tails.get(rank)
            if tail is None:
                tail = self.rank_tails[rank] = JsonlTail(path)
            for record in tail.read_new():
                if record.get("type") == "event":
                    self._add_event(record)
                    continue
                if record.get("type") != "record":
                    continue
                try:
                    step = int(record.get("step"))
                except (TypeError, ValueError):
                    continue
                self.pending_ranks.setdefault(step, {})[rank] = record
        world = self._world or (max(self.rank_tails) + 1 if self.rank_tails else 0)
        for step in sorted(list(self.pending_ranks)):
            records = self.pending_ranks[step]
            if world and set(records) >= set(range(world)):
                self._publish_fallback(step, records, complete=True)
            elif now - self._fallback_partial_at.get(step, now) > 30.0:
                self._fallback_partial_at.setdefault(step, now)
                if any(later > step for later in self.pending_ranks):
                    self._publish_fallback(step, records, complete=False)

    def _publish_fallback(self, step: int, records: Dict[int, Dict[str, Any]], complete: bool) -> None:
        values = list(records.values())
        flat = [_flatten_record(record) for record in values]
        sample = sum(_num(item.get("data/samples")) or 0.0 for item in flat)
        loss_values = [_num(item.get("loss/total")) for item in flat]
        finite_loss = [value for value in loss_values if value is not None]
        aggregated = {
            "type": "record",
            "schema": 1,
            "step": step,
            "rank": 0,
            "time": max((_num(record.get("time")) or 0.0) for record in values),
            "ranks": sorted(records),
            "world": len(records),
            "expected_world": self._world or len(records),
            "complete": complete,
            "data": {
                "samples": sample,
                "ar_tokens": sum(_num(item.get("data/ar_tokens")) or 0.0 for item in flat),
                "dm_tokens": sum(_num(item.get("data/dm_tokens")) or 0.0 for item in flat),
                "wait_ms": max((_num(item.get("data/wait_ms")) or 0.0) for item in flat),
                "vae_ms": max((_num(item.get("data/vae_ms")) or 0.0) for item in flat),
                "pack_ms": max((_num(item.get("data/pack_ms")) or 0.0) for item in flat),
            },
            "loss": {"total": statistics.fmean(finite_loss)} if finite_loss else {},
            "perf": {
                "step_ms": max((_num(item.get("perf/step_ms")) or 0.0) for item in flat),
                "forward_ms": max(
                    (_num(item.get("perf/forward_ms")) or _num(item.get("perf/fwd_ms")) or 0.0) for item in flat
                ),
                "backward_ms": max(
                    (_num(item.get("perf/backward_ms")) or _num(item.get("perf/bwd_ms")) or 0.0) for item in flat
                ),
                "optim_ms": max((_num(item.get("perf/optim_ms")) or 0.0) for item in flat),
                "skew": None,
            },
            "mem": {
                "device_used_gib": max((_num(item.get("mem/device_used_gib")) or 0.0) for item in flat),
                "peak_reserved_gib": max((_num(item.get("mem/peak_reserved_gib")) or 0.0) for item in flat),
                "allocated_gib": max((_num(item.get("mem/allocated_gib")) or 0.0) for item in flat),
                "device_total_gib": max((_num(item.get("mem/device_total_gib")) or 0.0) for item in flat),
            },
        }
        if not complete:
            aggregated["data"]["partial_samples"] = aggregated["data"].pop("samples", None)
        if self.series.append(aggregated):
            self._last_record = aggregated
        self.pending_ranks.pop(step, None)
        self._fallback_partial_at.pop(step, None)

    def _read_heartbeats(self) -> None:
        for path in self.ref.metrics_dir.glob("heartbeat-*.json"):
            try:
                rank = int(path.stem.split("-")[-1])
            except (ValueError, IndexError):
                continue
            reader = self.heartbeat_files.get(rank)
            if reader is None or reader.path != path:
                reader = self.heartbeat_files[rank] = JsonFile(path)
            reader.read()

    def _read_terminals(self) -> None:
        for path in self.ref.metrics_dir.glob("terminal-*.json"):
            try:
                rank = int(path.stem.split("-")[-1])
            except (ValueError, IndexError):
                continue
            reader = self.terminal_files.get(rank)
            if reader is None or reader.path != path:
                reader = self.terminal_files[rank] = JsonFile(path)
            reader.read()

    def _read_launchers(self) -> None:
        candidates: List[Path] = []
        for directory in (self.ref.path / "control", self.ref.path, self.ref.metrics_dir):
            if directory.is_dir():
                candidates.extend(sorted(directory.glob("launcher-*.json")))
        for path in candidates:
            reader = self.launcher_files.get(str(path))
            if reader is None:
                reader = self.launcher_files[str(path)] = JsonFile(path)
            reader.read()

    def _read_checkpoint_ledger(self, now: float) -> None:
        path = self.ref.metrics_dir / "checkpoints.jsonl"
        if path.is_file():
            if self.checkpoint_tail.truncated:
                self.checkpoints.clear()
                self.checkpoint_tail.truncated = False
            for record in self.checkpoint_tail.read_new():
                checkpoint_id = record.get("checkpoint_id") or record.get("id")
                if not checkpoint_id:
                    continue
                entry = self.checkpoints.setdefault(str(checkpoint_id), {"checkpoint_id": str(checkpoint_id)})
                entry.update({key: value for key, value in record.items() if key not in ("type", "schema")})
                if record.get("event") == "end":
                    entry["status"] = record.get("status") or "complete"
                elif record.get("event") == "begin" and "status" not in entry:
                    entry["status"] = "pending"
        if now - self._checkpoint_scan_at < 10.0:
            return
        self._checkpoint_scan_at = now
        self._scan_checkpoint_dirs()

    def _scan_checkpoint_dirs(self) -> None:
        """Derive checkpoints from the filesystem when no ledger exists (old runs, harness)."""
        try:
            entries = list(self.ref.path.iterdir())
        except OSError:
            entries = []
        for directory_name in ("training_state", "checkpoints"):
            directory = self.ref.path / directory_name
            if directory.is_dir():
                try:
                    entries.extend(directory.glob("checkpoint-*"))
                except OSError:
                    pass
        for path in entries:
            if not path.is_dir():
                continue
            if not (path.name.startswith("step-") or path.name.startswith("checkpoint-")):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            existing = next(
                (item for item in self.checkpoints.values() if item.get("path") == str(path)),
                None,
            )
            if existing is not None and existing.get("_mtime") == mtime:
                continue
            try:
                from examples.worldmodel.cosmos3_harness.checkpoint_ledger import measure_checkpoint
            except Exception:
                measure_checkpoint = None
            size = files = None
            source = "filesystem"
            if measure_checkpoint is not None:
                size, files, source = measure_checkpoint(path)
            if path.name.startswith("step-"):
                index = path / "model.safetensors.index.json"
                step = _step_from_name(path.name)
                entry = {
                    "checkpoint_id": f"filesystem-{path.name}",
                    "step": step,
                    "kind": "model_export",
                    "path": str(path),
                    "status": "complete" if index.is_file() else "incomplete",
                    "bytes": size,
                    "files": files,
                    "ended_at": mtime,
                    "started_at": mtime,
                    "source": source,
                    "_mtime": mtime,
                }
            else:
                complete = path / "complete.json"
                payload = read_json(complete, {}) if complete.is_file() else {}
                step = payload.get("step") if isinstance(payload, dict) else None
                entry = {
                    "checkpoint_id": f"filesystem-{path.name}",
                    "step": step if isinstance(step, int) else None,
                    "kind": "resumable_generation",
                    "path": str(path),
                    "status": "complete" if complete.is_file() else "incomplete",
                    "bytes": size,
                    "files": files,
                    "generation": path.name,
                    "ended_at": mtime,
                    "started_at": mtime,
                    "source": source,
                    "_mtime": mtime,
                }
            if existing is None:
                self.checkpoints[entry["checkpoint_id"]] = entry
            else:
                # The ledger owns the identity and step; the filesystem may fill gaps and correct a size that
                # was recorded with the wrong unit by an older writer.
                for key in ("status", "files", "source", "ended_at", "started_at", "generation"):
                    if existing.get(key) in (None, ""):
                        existing[key] = entry.get(key)
                if existing.get("source") == "index" and entry.get("source") not in (None, "index"):
                    existing["source"] = entry["source"]
                old_size = _num(existing.get("bytes"))
                new_size = _num(entry.get("bytes"))
                if new_size is not None and (old_size is None or new_size > 2 * max(old_size, 1.0)):
                    existing["bytes"] = new_size
                existing["_mtime"] = mtime

    # -- derived ------------------------------------------------------------------------------------------

    @property
    def world(self) -> int:
        return self._world or (max(self.rank_tails) + 1 if self.rank_tails else 0)

    @property
    def max_steps(self) -> Optional[int]:
        return self._max_steps

    @property
    def last_record(self) -> Optional[Dict[str, Any]]:
        return self._last_record

    def latest_heartbeats(self) -> Dict[int, Dict[str, Any]]:
        result: Dict[int, Dict[str, Any]] = {}
        for rank, reader in self.heartbeat_files.items():
            value = reader.read()
            if isinstance(value, dict):
                value = dict(value)
                value["_file_age_s"] = file_age(reader.path)
                value["_rank"] = rank
                value["_path"] = str(reader.path)
                result[rank] = value
        return result

    def latest_terminals(self) -> Dict[int, Dict[str, Any]]:
        result: Dict[int, Dict[str, Any]] = {}
        for rank, reader in self.terminal_files.items():
            value = reader.read()
            if isinstance(value, dict):
                result[rank] = value
        return result

    def latest_launchers(self) -> List[Dict[str, Any]]:
        result = []
        for reader in self.launcher_files.values():
            value = reader.read()
            if isinstance(value, dict):
                result.append(value)
        return result

    def issues(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        now = time.time() if now is None else now
        world = self.world
        heartbeats = self.latest_heartbeats()
        terminals = self.latest_terminals()
        launchers = self.latest_launchers()
        issues: List[Dict[str, Any]] = []

        def add(kind: str, message: str, severity: str = "warn", **extra: Any) -> None:
            item = {"kind": kind, "message": message, "severity": severity}
            item.update({key: value for key, value in extra.items() if value is not None})
            issues.append(item)

        completed = self._completed(world, terminals, launchers)
        if not completed and not heartbeats and not terminals and self.last_record is not None:
            age = self.age_s(now)
            legacy_grace = max(1800.0, 3 * (self._median_step_s() or 0.0))
            if age > legacy_grace:
                add("stale_run", f"no heartbeat or terminal files; last update {age:.0f}s ago", "warn")
        failed_ranks = [rank for rank, terminal in terminals.items() if not terminal.get("completed")]
        failed_launchers = [launcher for launcher in launchers if launcher.get("state") == "failed"]
        if failed_ranks or failed_launchers:
            add("process_failure", f"failed ranks={sorted(failed_ranks)} launchers={len(failed_launchers)}", "critical")
        for rank, terminal in terminals.items():
            if terminal.get("cleanup_warning"):
                add("worker_shutdown_warning", str(terminal["cleanup_warning"]), "warn", rank=rank)

        if not completed and not failed_ranks and not failed_launchers:
            expected = range(world) if world else sorted(heartbeats)
            for rank in expected:
                heartbeat = heartbeats.get(rank)
                if heartbeat is None:
                    if launchers and now - min((_num(item.get("start_time")) or now) for item in launchers) > 900:
                        add("missing_heartbeat", "no heartbeat 15 minutes after launch", "warn", rank=rank)
                    continue
                age = _num(heartbeat.get("_file_age_s"))
                if age is not None and age > self.stale_seconds:
                    add("stale_heartbeat", f"heartbeat file is {age:.0f}s old", "critical", rank=rank, age_s=age)
                    continue
                writer_now = _num(heartbeat.get("time"))
                progress = _num(heartbeat.get("progress_time")) or writer_now
                phase = heartbeat.get("phase")
                stall = self._adaptive_stall()
                if phase == "checkpoint" or "checkpoint" in str(phase):
                    stall = max(stall, 900.0)
                if writer_now is not None and progress is not None and writer_now - progress > stall:
                    add(
                        "stalled_progress",
                        f"no progress for {writer_now - progress:.0f}s in phase {phase}",
                        "critical",
                        rank=rank,
                        phase=phase,
                    )
                if heartbeat.get("error"):
                    add("heartbeat_sampling_error", str(heartbeat["error"]), "warn", rank=rank)

        # A launcher can outlive the heartbeat thread; the collector's process table is independent evidence.
        latest_system = {item.get("hostname"): item for item in self.system.latest()}
        for rank, heartbeat in heartbeats.items():
            host = heartbeat.get("hostname")
            system = latest_system.get(host)
            if not system:
                continue
            pids = {process.get("pid") for process in system.get("processes") or []}
            pid = heartbeat.get("pid")
            if pid and pid not in pids and not completed:
                add(
                    "rank_process_missing",
                    f"rank {rank} pid {pid} is not in the process table",
                    "error",
                    rank=rank,
                    host=host,
                )
        for launcher in launchers:
            if launcher.get("state") != "running":
                continue
            host = launcher.get("hostname") or launcher.get("node")
            system = latest_system.get(host)
            if system and not system.get("processes"):
                add(
                    "launcher_process_missing",
                    f"launcher on {host} is running but no training process is visible",
                    "error",
                    host=host,
                )

        if launchers and any(item.get("state") == "completed" for item in launchers):
            if world and len(terminals) < world and not failed_ranks:
                add("incomplete_terminal_ranks", f"{len(terminals)}/{world} terminal files", "error")

        # Checkpoints.
        latest_size = None
        for entry in sorted(self.checkpoints.values(), key=lambda item: item.get("step") or 0):
            status = entry.get("status")
            if status == "failed":
                add(
                    "checkpoint_failed",
                    f"checkpoint at step {entry.get('step')} failed: {entry.get('error')}",
                    "critical",
                    step=entry.get("step"),
                )
            if status == "pending":
                started = _num(entry.get("started_at")) or 0.0
                if started and now - started > CHECKPOINT_PENDING_SECONDS:
                    add(
                        "checkpoint_missing",
                        f"checkpoint at step {entry.get('step')} has no end record",
                        "error",
                        step=entry.get("step"),
                    )
            size = _num(entry.get("bytes"))
            if size and latest_size and (size < 0.5 * latest_size or size > 2.0 * latest_size):
                add(
                    "checkpoint_size_anomaly",
                    f"checkpoint size {size} differs from previous {latest_size}",
                    "warn",
                    step=entry.get("step"),
                )
            if size:
                latest_size = size
        durations = [
            _num(entry.get("duration_ms"))
            for entry in self.checkpoints.values()
            if _num(entry.get("duration_ms")) is not None
        ]
        if len(durations) >= 3:
            median = statistics.median(durations)
            for entry in self.checkpoints.values():
                duration = _num(entry.get("duration_ms"))
                if duration is not None and duration > max(120000.0, 3 * median):
                    add("checkpoint_slow", f"checkpoint took {duration / 1e3:.1f}s", "warn", step=entry.get("step"))

        # System rules.
        for host, system in latest_system.items():
            host_info = system.get("host") or {}
            for path, disk in (host_info.get("disk_free_gib") or {}).items():
                if isinstance(disk, dict):
                    free = _num(disk.get("free_gib"))
                    total = _num(disk.get("total_gib"))
                else:
                    free, total = _num(disk), None
                if free is not None and total:
                    threshold = max(10.0, 0.1 * total)
                    if free < threshold:
                        add("disk_low", f"{host}:{path} has {free:.0f} GiB free", "critical", host=host)
            shm = host_info.get("shm") or {}
            shm_free, shm_total = _num(shm.get("free_gib")), _num(shm.get("total_gib"))
            if shm_free is not None and shm_total and shm_free < 0.1 * shm_total:
                add(
                    "shm_pressure",
                    f"{host}: /dev/shm is {100 * (1 - shm_free / shm_total):.0f}% full",
                    "warn",
                    host=host,
                )
            memory = host_info.get("memory") or {}
            available, total = _num(memory.get("available_gib")), _num(memory.get("total_gib"))
            if available is not None and total and available < 0.1 * total:
                add("host_memory_low", f"{host}: {available:.0f}/{total:.0f} GiB available", "warn", host=host)
            for device in system.get("devices") or []:
                used, capacity = _num(device.get("hbm_used_mib")), _num(device.get("hbm_total_mib"))
                if used is not None and capacity:
                    fraction = used / capacity
                    index = device.get("index")
                    if fraction > 0.95:
                        add("hbm_critical", f"{host} NPU {index}: HBM {fraction:.0%}", "error", host=host, device=index)
                    elif fraction > 0.90:
                        add("hbm_pressure", f"{host} NPU {index}: HBM {fraction:.0%}", "warn", host=host, device=index)
                temperature = _num(device.get("temperature_c"))
                if temperature is not None:
                    if temperature > 95:
                        add(
                            "temperature_critical",
                            f"{host} NPU {device.get('index')}: {temperature:.0f}C",
                            "error",
                            host=host,
                            device=device.get("index"),
                        )
                    elif temperature > 85:
                        add(
                            "temperature_high",
                            f"{host} NPU {device.get('index')}: {temperature:.0f}C",
                            "warn",
                            host=host,
                            device=device.get("index"),
                        )

        # Data not advancing while the workers are alive: the loss chart is flat but nothing crashed.
        if not completed and not failed_ranks and not failed_launchers and heartbeats and self.last_record is not None:
            last_time = _num(self.last_record.get("time"))
            median = self._median_step_s()
            if last_time is not None and median and now - last_time > max(3 * median, 60.0):
                add(
                    "data_not_advancing",
                    f"no new aggregated step for {now - last_time:.0f}s while heartbeats are live",
                    "warn",
                )

        if not self._manifest:
            add("missing_manifest", "no run.json or config.json was found", "warn")
        return issues

    def _adaptive_stall(self) -> float:
        median = self._median_step_s()
        return max(self.stall_seconds, 4 * median) if median else self.stall_seconds

    def _median_step_s(self) -> Optional[float]:
        values = [value for value in self.series.series["perf/step_s"].y[-20:] if value is not None]
        return statistics.median(values) if values else None

    def _completed(self, world: int, terminals: Dict[int, Dict[str, Any]], launchers: List[Dict[str, Any]]) -> bool:
        if self.summary and self.summary.get("completed") is True:
            return True
        if not terminals and self._max_steps and self.last_record is not None:
            # A run from before the terminal file existed can still be recognised as finished: it reached the
            # declared number of optimizer steps and has not written anything for a while.
            step = int(_num(self.last_record.get("step")) or 0)
            if step >= self._max_steps and self.age_s() > max(300.0, 3 * (self._median_step_s() or 0.0)):
                return True
        if not world or len(terminals) < world:
            return False
        if not all(terminal.get("completed") for terminal in terminals.values()):
            return False
        if launchers and not all(launcher.get("state") == "completed" for launcher in launchers):
            return False
        return True

    def state(self, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        world = self.world
        heartbeats = self.latest_heartbeats()
        terminals = self.latest_terminals()
        launchers = self.latest_launchers()
        if any(not terminal.get("completed") for terminal in terminals.values()) or any(
            launcher.get("state") == "failed" for launcher in launchers
        ):
            return "failed"
        if self._completed(world, terminals, launchers):
            return "completed"
        issues = self.issues(now)
        if any(item.get("severity") in ("warn", "error", "critical") for item in issues):
            return "attention"
        if heartbeats or self.last_record is not None:
            return "running"
        return "initializing"

    def updated_at(self) -> float:
        # Deliberately not the server's refresh time: a dead writer must look old even while the monitor keeps
        # polling.  File mtimes are the cross-host clock-free signal; the record timestamp is a fallback.
        candidates = []
        for path in (self.ref.metrics_dir, self.ref.path / "control", self.ref.path):
            try:
                candidates.append(path.stat().st_mtime)
            except OSError:
                pass
        if self.last_record:
            candidates.append(_num(self.last_record.get("time")) or 0.0)
        return max(candidates) if candidates else 0.0

    def age_s(self, now: Optional[float] = None) -> float:
        return max(0.0, (time.time() if now is None else now) - self.updated_at())

    # -- public views -------------------------------------------------------------------------------------

    def snapshot(self, now: Optional[float] = None, *, include_manifest: bool = False) -> Dict[str, Any]:
        now = time.time() if now is None else now
        with self.lock:
            return self._snapshot_locked(now, include_manifest=include_manifest)

    def _snapshot_locked(self, now: float, *, include_manifest: bool) -> Dict[str, Any]:
        heartbeats = self.latest_heartbeats()
        terminals = self.latest_terminals()
        issues = self.issues(now)
        last = self.last_record or {}
        flat = _flatten_record(last)
        step = int(flat.get("step") or last.get("step") or 0)
        max_steps = self.max_steps
        step_s = self.series.latest_value("perf/step_s")
        median = self._median_step_s()
        eta_s = (max_steps - step) * median if max_steps and median and step < max_steps else None
        loss = self.series.latest_value("loss/total")
        device_used = self.series.latest_value("mem/device_used_gib")
        device_total = self.series.latest_value("mem/device_total_gib")
        config = self._manifest.get("config") if isinstance(self._manifest.get("config"), dict) else {}
        payload: Dict[str, Any] = {
            "run_id": str(self._manifest.get("run_id") or config.get("run_id") or self.ref.display_name),
            "display_name": str(
                self._manifest.get("run_name") or self._manifest.get("run_id") or self.ref.display_name
            ),
            "token": self.ref.token,
            "dir": str(self.ref.path),
            "metrics_dir": str(self.ref.metrics_dir),
            "layout": self.ref.layout,
            "state": self.state(now),
            "age_s": self.age_s(now),
            "stale": self.age_s(now) > 3 * self.stale_seconds,
            "updated_at": self.updated_at(),
            "started_at": self._started_at,
            "world": self.world,
            "live_ranks": sum(
                1
                for heartbeat in heartbeats.values()
                if (_num(heartbeat.get("_file_age_s")) or self.stale_seconds + 1) <= self.stale_seconds
            ),
            "terminal_ranks": len(terminals),
            "step": step,
            "max_steps": max_steps,
            "progress": (step / max_steps) if max_steps else None,
            "eta_s": eta_s,
            "epoch": self.series.latest_value("data/epoch"),
            "last": {
                "step": step,
                "time": _num(last.get("time")),
                "loss": loss,
                "vision": self.series.latest_value("loss/vision"),
                "action": self.series.latest_value("loss/action"),
                "lr": self.series.latest_value("optim/lr"),
                "grad_norm": self.series.latest_value("optim/grad_norm"),
                "step_s": step_s,
                "samples_per_s": self.series.latest_value("perf/samples_per_s"),
                "mfu": self.series.latest_value("perf/mfu"),
                "device_used_gib": device_used,
                "device_total_gib": device_total,
                "hbm_fraction": (device_used / device_total) if device_used and device_total else None,
                "skew": self.series.latest_value("perf/skew"),
                "complete": last.get("complete"),
            },
            "issues": issues,
            "warnings": self._errors
            + [f"{reader.path}: {reader.error}" for reader in (self.manifest, self.config) if reader and reader.error],
            "hosts": self._hosts(),
            "topology": derive_topology(self._manifest, self.world),
            "checkpoints": {
                "count": len(self.checkpoints),
                "last": self._last_checkpoint(),
                "bytes": sum(_num(entry.get("bytes")) or 0.0 for entry in self.checkpoints.values()),
            },
            "sparkline": {
                "loss": [value for value in self.series.series["loss/total"].y[-60:] if value is not None],
                "step_s": [value for value in self.series.series["perf/step_s"].y[-60:] if value is not None],
            },
        }
        if include_manifest:
            payload["manifest"] = self._manifest
        return payload

    def _last_checkpoint(self) -> Optional[Dict[str, Any]]:
        if not self.checkpoints:
            return None
        return max(
            self.checkpoints.values(),
            key=lambda item: (_num(item.get("ended_at")) or 0.0, _num(item.get("step")) or 0.0),
        )

    def _hosts(self) -> List[Dict[str, Any]]:
        hosts: Dict[str, Dict[str, Any]] = {}
        for row in derive_topology(self._manifest, self.world):
            host = row.get("hostname")
            if not host:
                continue
            entry = hosts.setdefault(host, {"hostname": host, "ranks": [], "derived": bool(row.get("derived"))})
            entry["ranks"].append(row.get("rank"))
        for host, entry in hosts.items():
            entry["ranks"] = sorted(rank for rank in entry["ranks"] if rank is not None)
        return list(hosts.values())

    # -- views --------------------------------------------------------------------------------------------

    def events_page(
        self, severity: Optional[str] = None, kind: Optional[str] = None, limit: int = 200
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 2000))
        result = []
        for event in reversed(self.events):
            if severity and event.get("severity") != severity:
                continue
            if kind and event.get("kind") != kind:
                continue
            result.append(event)
            if len(result) >= limit:
                break
        return result

    def _add_event(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    def checkpoints_view(self) -> Dict[str, Any]:
        rows = sorted(
            self.checkpoints.values(),
            key=lambda item: (_num(item.get("step")) or 0.0, _num(item.get("ended_at")) or 0.0),
        )
        totals = {
            "count": len(rows),
            "bytes": sum(_num(row.get("bytes")) or 0.0 for row in rows),
            "failed": sum(1 for row in rows if row.get("status") == "failed"),
            "pending": sum(1 for row in rows if row.get("status") == "pending"),
        }
        return {"checkpoints": rows, "totals": totals}

    def ranks_view(self) -> Dict[str, Any]:
        heartbeats = self.latest_heartbeats()
        terminals = self.latest_terminals()
        world = self.world or (max(heartbeats) + 1 if heartbeats else 0)
        rows = []
        for rank in range(world):
            heartbeat = heartbeats.get(rank)
            terminal = terminals.get(rank)
            row = {
                "rank": rank,
                "heartbeat": heartbeat,
                "terminal": terminal,
                "alive": bool(heartbeat and (_num(heartbeat.get("_file_age_s")) or 1e9) <= self.stale_seconds),
            }
            rows.append(row)
        return {"world": world, "ranks": rows}


def _args_to_dict(args: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not isinstance(args, list):
        return result
    index = 0
    while index < len(args):
        item = str(args[index])
        if item.startswith("--"):
            key = item[2:].replace("-", "_")
            if index + 1 < len(args) and not str(args[index + 1]).startswith("--"):
                result[key] = args[index + 1]
                index += 2
                continue
            result[key] = True
        index += 1
    return result


def _step_from_name(name: str) -> Optional[int]:
    _, _, suffix = name.partition("-")
    try:
        return int(suffix)
    except ValueError:
        return None
