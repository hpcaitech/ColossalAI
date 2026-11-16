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

from examples.worldmodel.cosmos3_harness.training_metrics import aggregate_losses

from .control import STATUS_FRESH_SECONDS, checkpoint_root, latest_generation
from .discovery import RunRef, derive_topology, load_manifest
from .discovery import max_steps as manifest_max_steps
from .discovery import world_size
from .readers import JsonFile, JsonlTail, file_age, read_json
from .series import SeriesStore, ThroughputStore, normalise_loss_record, stage_metrics
from .system import SystemHistory, read_system_files

STALE_SECONDS = 15.0
STALL_SECONDS = 300.0
CHECKPOINT_PENDING_SECONDS = 900.0
#: A rank file at or below this size is read from the start on first sight; a larger one only from its tail.
INITIAL_RANK_BYTES = 8 * 1024 * 1024
TAIL_RANK_BYTES = 512 * 1024


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


#: An end-of-run record written by a build without ``run_started_at`` is judged by its timestamp; this grace
#: covers ordinary clock skew between a rank and the rank that wrote the manifest.
_CLOCK_GRACE_SECONDS = 60.0


def _belongs_to_current_run(item: Dict[str, Any], started_at: Optional[float]) -> bool:
    """Whether a terminal/summary record describes the *current* attempt in this run directory.

    ``run_started_at`` is copied out of the manifest by both sides, so comparing it is exact and needs no
    clock agreement.  Records from older builds fall back to their timestamp.
    """
    if started_at is None:
        return True
    stamp = item.get("run_started_at")
    if stamp is not None:
        try:
            return abs(float(stamp) - float(started_at)) <= 1e-3
        except (TypeError, ValueError):
            return True
    when = _num(item.get("time")) if item.get("time") is not None else _num(item.get("ended_at"))
    if when is None:
        return True
    return when >= float(started_at) - _CLOCK_GRACE_SECONDS


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
        self.control_request_file = JsonFile(ref.path / "control" / "request.json")
        self.control_status_file = JsonFile(ref.path / "control" / "status.json")
        self.series = SeriesStore()
        self.throughput = ThroughputStore()
        self.system = SystemHistory()
        self.events: Deque[Dict[str, Any]] = deque(maxlen=4000)
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.pending_ranks: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self.partial_aggregates: Dict[int, Dict[str, Any]] = {}
        self._partial_seen_at: Dict[int, float] = {}
        self._partial_signature: Dict[int, tuple] = {}
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
            self.partial_aggregates.clear()
            self._partial_seen_at.clear()
            self._partial_signature.clear()
            self.events.clear()
            self.checkpoints.clear()
            self.checkpoint_tail.reset()
            self.summary = None
            self._last_record = None
            self.throughput = ThroughputStore()
            self.system = SystemHistory()
        self._run_id = run_id or self._run_id
        self._world = world_size(self._manifest) or self._world
        self._max_steps = manifest_max_steps(self._manifest) or self._max_steps
        started_at = _num(self._manifest.get("started_at"))
        if started_at is not None and self._started_at is not None and abs(started_at - self._started_at) > 1e-3:
            # The same directory started a new attempt (a resume).  Its summary.json record and terminal
            # files describe the previous one until they are rewritten, so drop the summary now.
            self.summary = None
        self._started_at = started_at or self._started_at
        self.series.set_world(self._world)
        self._errors = []
        try:
            self._read_aggregated()
            self._read_rank_files(now)
            self._read_heartbeats()
            self._read_terminals()
            self._read_launchers()
            self._read_control()
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
            self.partial_aggregates.clear()
            self._partial_seen_at.clear()
            self._partial_signature.clear()
            for tail in self.rank_tails.values():
                tail.reset()
            self.aggregated.truncated = False
        for record in self.aggregated.read_new():
            kind = record.get("type")
            if kind == "record":
                record = normalise_loss_record(record, self._manifest)
                display = self._estimate_global_record(record) if record.get("complete") is False else record
                if self.series.append(display):
                    self._last_record = display
                    if record.get("complete") is False:
                        try:
                            step = int(record.get("step"))
                        except (TypeError, ValueError):
                            step = None
                        if step is not None:
                            self.partial_aggregates[step] = record
                            self._partial_seen_at.setdefault(step, time.time())
            elif kind == "event":
                self._add_event(record)
            elif kind == "summary":
                payload = record.get("summary") or record
                if _belongs_to_current_run(record, self._started_at) and _belongs_to_current_run(
                    payload, self._started_at
                ):
                    self.summary = payload

    def _configured_samples_per_rank(self) -> Optional[float]:
        args = self._manifest.get("args")
        if isinstance(args, list):
            args = _args_to_dict(args)
        if not isinstance(args, dict):
            config = self._manifest.get("config") if isinstance(self._manifest.get("config"), dict) else {}
            args = _args_to_dict(config.get("args"))
        per_rank = _num(args.get("samples_per_rank"))
        if per_rank is None or per_rank <= 0:
            return None
        accum = _num(args.get("grad_accum")) or 1.0
        return per_rank * accum

    def _estimate_global_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Scale a partial aggregate to the expected world when the per-rank batch is known and uniform.

        Rank 0 writes ``partial_samples = arrived_ranks x samples_per_rank``.  If that ratio matches the launch
        arguments, the missing ranks are known to contribute the same amount, so the panel can show the global
        batch immediately instead of waiting for the mirror.  The rank-file correction later replaces this with
        measured token/stage totals and marks the step complete.
        """
        data = record.get("data") if isinstance(record.get("data"), dict) else {}
        raw = _num(data.get("partial_samples"))
        if raw is None:
            raw = _num(data.get("samples"))
        arrived = len(record.get("ranks") or []) or int(_num(record.get("world")) or 0)
        expected = int(_num(record.get("expected_world")) or self._world or 0)
        per_rank = self._configured_samples_per_rank()
        if not raw or not arrived or not expected or expected <= arrived or not per_rank:
            return record
        ratio = raw / arrived
        if abs(ratio - per_rank) > max(0.01, per_rank * 0.01):
            return record
        corrected = dict(record)
        corrected["data"] = dict(data)
        corrected["data"]["partial_samples"] = raw
        corrected["data"]["samples"] = ratio * expected
        scale = expected / arrived
        for key in ("ar_tokens", "dm_tokens"):
            value = _num(corrected["data"].get(key))
            if value is not None:
                corrected["data"][key] = value * scale
        corrected["world"] = expected
        corrected["expected_world"] = expected
        corrected["reporting_world"] = arrived
        corrected["samples_estimated"] = True
        corrected["complete"] = False
        corrected["source"] = corrected.get("source") or "configured_batch"
        return corrected

    def _read_rank_files(self, now: float) -> None:
        """Tail every rank file: it is both the legacy aggregate source and the per-rank throughput source.

        Only the last 512 KiB of each rank file is read when the monitor first sees it, so opening a
        100k-step run does not parse the whole history.  Live tails continue from there.
        """
        paths = sorted(self.ref.metrics_dir.glob("rank-*.jsonl"))
        if not paths:
            return
        rank_hosts = self._rank_hosts()
        for path in paths:
            try:
                rank = int(path.stem.split("-")[-1])
            except (ValueError, IndexError):
                continue
            tail = self.rank_tails.get(rank)
            if tail is None:
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                # Read a small rank file from the start: a restart then keeps the run's whole per-rank history,
                # which the step drawer and the rank heatmap need (2.4 MB ≈ 2.5k steps).  A long run's file is
                # only tailed, so a restart does not turn into a many-hundred-megabyte parse.
                offset = 0 if size <= INITIAL_RANK_BYTES else max(0, size - TAIL_RANK_BYTES)
                tail = self.rank_tails[rank] = JsonlTail(path, offset=offset)
            for record in tail.read_new():
                if record.get("type") == "event":
                    if not self._aggregated_seen:
                        self._add_event(record)
                    continue
                if record.get("type") != "record":
                    continue
                record = normalise_loss_record(record, self._manifest)
                try:
                    step = int(record.get("step"))
                except (TypeError, ValueError):
                    continue
                if not self._aggregated_seen:
                    self.pending_ranks.setdefault(step, {})[rank] = record
                samples = _num(record.get("data", {}).get("samples"))
                if samples is None:
                    samples = _num(record.get("data", {}).get("partial_samples"))
                step_ms = _num(record.get("perf", {}).get("step_ms"))
                if samples is not None and step_ms:
                    self.throughput.ingest(
                        rank, rank_hosts.get(rank, "unknown"), step, samples, step_ms / 1e3, now, record
                    )
        world = self._world or (max(self.rank_tails) + 1 if self.rank_tails else 0)
        host_ranks = self._host_rank_map()
        self.throughput.finalize_pending(host_ranks, world, now)
        self._correct_partial_aggregates(now)
        if self._aggregated_seen:
            return
        for step in sorted(list(self.pending_ranks)):
            records = self.pending_ranks[step]
            if world and set(records) >= set(range(world)):
                self._publish_fallback(step, records, complete=True)
            elif now - self._fallback_partial_at.get(step, now) > 30.0:
                self._fallback_partial_at.setdefault(step, now)
                if any(later > step for later in self.pending_ranks):
                    self._publish_fallback(step, records, complete=False)

    def _correct_partial_aggregates(self, now: float) -> None:
        """Rewrite rank 0's partial per-step record once the mirror has the other nodes' rank records.

        The original aggregate is already on the chart; ``SeriesStore.update`` replaces that same point, so a
        late rank file does not create an out-of-order step or a duplicate sample count.
        """
        if not self.partial_aggregates:
            return
        for step in sorted(list(self.partial_aggregates)):
            record = self.partial_aggregates.get(step)
            if not isinstance(record, dict):
                self.partial_aggregates.pop(step, None)
                continue
            entries = self.throughput.step_ranks.get(step) or {}
            expected = int(_num(record.get("expected_world")) or self._world or 0)
            age = now - self._partial_seen_at.get(step, now)
            if not entries:
                if age > 300.0:
                    self.partial_aggregates.pop(step, None)
                    self._partial_seen_at.pop(step, None)
                    self._partial_signature.pop(step, None)
                continue
            complete = not expected or len(entries) >= expected
            if not complete and age < 30.0:
                continue
            if not complete:
                existing = self.series.record_for_step(step)
                if isinstance(existing, dict) and existing.get("samples_estimated"):
                    # The configured per-rank batch already gives the exact global count; replacing it with a
                    # smaller sum of whichever ranks arrived would reintroduce the six-times-too-small view.
                    if age > 300.0:
                        self.partial_aggregates.pop(step, None)
                        self._partial_seen_at.pop(step, None)
                        self._partial_signature.pop(step, None)
                    continue
            samples = sum(_num(entry.get("samples")) or 0.0 for entry in entries.values())
            signature = (len(entries), round(samples, 3), complete)
            if self._partial_signature.get(step) == signature:
                if complete:
                    self.partial_aggregates.pop(step, None)
                    self._partial_seen_at.pop(step, None)
                    self._partial_signature.pop(step, None)
                continue
            step_s = max((_num(entry.get("step_s")) or 0.0) for entry in entries.values())
            if samples <= 0 or step_s <= 0:
                continue
            corrected = dict(record)
            corrected["data"] = dict(record.get("data") or {})
            corrected["perf"] = dict(record.get("perf") or {})
            corrected["mem"] = dict(record.get("mem") or {})
            corrected["data"]["samples"] = samples
            corrected["data"].pop("partial_samples", None)
            corrected["perf"]["step_ms"] = step_s * 1e3
            corrected["world"] = self._world or expected or len(entries)
            corrected["expected_world"] = expected or self._world or len(entries)
            corrected["reporting_world"] = len(entries)
            corrected["ranks"] = sorted(entries)
            corrected["complete"] = complete
            corrected["aggregate_from_ranks"] = True
            corrected["aggregate_samples"] = _num((record.get("data") or {}).get("samples"))
            corrected["source"] = "rank_files" if complete else "rank_files_partial"
            loss = aggregate_losses(
                {"loss": entry.get("loss"), "data": {"samples": entry["samples"]}} for entry in entries.values()
            )
            if loss:
                corrected["loss"] = loss
            step_times = [_num(entry.get("step_s")) for entry in entries.values()]
            step_times = [value for value in step_times if value is not None and value > 0]
            if len(step_times) >= 2:
                mean_step = sum(step_times) / len(step_times)
                if mean_step > 0:
                    corrected["perf"]["skew"] = max(step_times) / mean_step
            for key in ("ar_tokens", "dm_tokens"):
                values = [_num(entry.get(key)) for entry in entries.values() if _num(entry.get(key)) is not None]
                if values:
                    corrected["data"][key] = sum(values)
            for key in ("wait_ms", "vae_ms", "pack_ms"):
                values = [_num(entry.get(key)) for entry in entries.values() if _num(entry.get(key)) is not None]
                if values:
                    corrected["data"][key] = max(values)
            for key in ("forward_ms", "backward_ms", "optim_ms"):
                values = [_num(entry.get(key)) for entry in entries.values() if _num(entry.get(key)) is not None]
                if values:
                    corrected["perf"][key] = max(values)
            for key in ("device_used_gib", "peak_reserved_gib", "allocated_gib", "device_total_gib"):
                values = [_num(entry.get(key)) for entry in entries.values() if _num(entry.get(key)) is not None]
                if values:
                    corrected["mem"][key] = max(values)
            for key in ("flops", "flops_forward", "flops_backward", "flops_vae"):
                values = [_num(entry.get(key)) for entry in entries.values() if _num(entry.get(key)) is not None]
                if values:
                    corrected["perf"][key] = sum(values)
            for key in ("mfu", "mfu_forward", "mfu_backward", "mfu_vae"):
                values = [_num(entry.get(key)) for entry in entries.values() if _num(entry.get(key)) is not None]
                if values:
                    corrected["perf"][key] = statistics.fmean(values)
            for key in ("peak_tflops_per_npu",):
                values = [_num(entry.get(key)) for entry in entries.values() if _num(entry.get(key)) is not None]
                if values:
                    corrected["perf"][key] = values[0]
            modes = [entry.get("checkpointing_mode") for entry in entries.values() if entry.get("checkpointing_mode")]
            if modes:
                corrected["perf"]["checkpointing_mode"] = str(modes[0])
            total_flops = _num(corrected["perf"].get("flops"))
            if total_flops and step_s > 0:
                corrected["perf"]["tflops"] = total_flops / step_s / 1e12
            if self.series.update(corrected):
                self._last_record = self.series.last()
            self._partial_signature[step] = signature
            if complete or age > 300.0:
                self.partial_aggregates.pop(step, None)
                self._partial_seen_at.pop(step, None)
                self._partial_signature.pop(step, None)

    def _host_rank_map(self) -> Dict[str, List[int]]:
        host_ranks: Dict[str, List[int]] = {}
        for row in derive_topology(self._manifest, self.world):
            host = row.get("hostname")
            rank = row.get("rank")
            if host is not None and rank is not None:
                host_ranks.setdefault(str(host), []).append(int(rank))
        for rank, heartbeat in self.latest_heartbeats().items():
            host = heartbeat.get("hostname")
            if host and all(rank not in ranks for ranks in host_ranks.values()):
                host_ranks.setdefault(str(host), []).append(int(rank))
        for ranks in host_ranks.values():
            ranks.sort()
        return host_ranks

    def _rank_hosts(self) -> Dict[int, str]:
        return {rank: host for host, ranks in self._host_rank_map().items() for rank in ranks}

    def throughput_series(self, source: str, x_axis: str = "step", max_points: int = 5000) -> Dict[str, Any]:
        data = self.throughput.series("nodes" if source == "nodes" else "ranks", max_points)
        if x_axis != "time":
            return data
        times = {int(record.get("step", -1)): _num(record.get("time")) for record in self.series.records}
        values = [value for value in times.values() if value is not None]
        if not values:
            return data
        base = min(values)
        for item in data.values():
            xs, ys = [], []
            for x, y in zip(item.get("x", []), item.get("y", [])):
                timestamp = times.get(int(x))
                if timestamp is None:
                    continue
                xs.append(timestamp - base)
                ys.append(y)
            item["x"], item["y"] = xs, ys
        return data

    def _publish_fallback(self, step: int, records: Dict[int, Dict[str, Any]], complete: bool) -> None:
        values = list(records.values())
        flat = [_flatten_record(record) for record in values]
        sample = sum(_num(item.get("data/samples")) or 0.0 for item in flat)
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
            "loss": aggregate_losses(values),
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

    def _read_control(self) -> None:
        """The stop/resume channel: what the dashboard asked for and what the loop last reported."""
        self.control_request_file.read()
        self.control_status_file.read()

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
            # A terminal file from the previous attempt in this directory survives a resume; only the
            # files stamped with this attempt's start (or, for older writers, newer than it) count.
            if isinstance(value, dict) and _belongs_to_current_run(value, self._started_at):
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
        failed_ranks = [
            rank
            for rank, terminal in terminals.items()
            if not terminal.get("completed") and not terminal.get("stopped")
        ]
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
            host_ranks = self._host_rank_map()
            launch_reference = self._started_at or (
                min((_num(item.get("time")) or now) for item in heartbeats.values()) if heartbeats else None
            )
            if launch_reference and now - launch_reference > max(120.0, self.stall_seconds):
                for host, ranks in sorted(host_ranks.items()):
                    if ranks and not any(rank in heartbeats for rank in ranks):
                        add(
                            "node_no_heartbeat",
                            f"no heartbeat from any of {len(ranks)} ranks on {host}",
                            "error",
                            host=host,
                        )

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
        if (
            terminals
            and len(terminals) >= max(1, world)
            and all(terminal.get("stopped") for terminal in terminals.values())
        ):
            # A deliberate stop: every rank exited after the same step with a resumable generation,
            # so this is a finished run, not a failure.
            return "stopped"
        if any(not terminal.get("completed") and not terminal.get("stopped") for terminal in terminals.values()) or any(
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
        # The run root is intentionally excluded: a mirror writes its bookkeeping there every few seconds, so
        # including it would keep a dead run looking fresh forever.  ``metrics_dir`` is the data plane.
        candidates = []
        for path in (self.ref.metrics_dir, self.ref.path / "control"):
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
        coverage = self.coverage(heartbeats, terminals)
        issues = self.issues(now)
        last = self.last_record or {}
        flat = _flatten_record(last)
        step = int(flat.get("step") or last.get("step") or 0)
        max_steps = self.max_steps
        step_s = self.series.latest_value("perf/step_s")
        median = self._median_step_s()
        recent_tp = [value for value in self.series.series["perf/samples_per_s"].y[-20:] if value and value > 0]
        recent_batch = [value for value in self.series.series["data/samples"].y[-20:] if value and value > 0]
        recent_throughput = statistics.median(recent_tp) if recent_tp else None
        recent_batch_median = statistics.median(recent_batch) if recent_batch else None
        partial_aggregate = last.get("complete") is False
        if partial_aggregate or recent_throughput is None or recent_batch_median is None:
            monitor_throughput, monitor_batch = self.throughput.recent_totals(self.world)
            if partial_aggregate or recent_throughput is None:
                recent_throughput = monitor_throughput or recent_throughput
            if partial_aggregate or recent_batch_median is None:
                recent_batch_median = monitor_batch or recent_batch_median
        remaining_steps = max(0, max_steps - step) if max_steps else 0
        run_state = self.state(now)
        # A stopped run is finished like a completed one: it has nothing left to estimate, and its elapsed
        # time must stop at the terminal file instead of growing while the dashboard keeps polling.
        completed_now = run_state in ("completed", "stopped")
        eta_s = None
        eta_source = None
        if completed_now:
            eta_s, eta_source = 0.0, run_state
        elif max_steps and remaining_steps:
            if recent_throughput and recent_batch_median:
                eta_s = remaining_steps * recent_batch_median / recent_throughput
                eta_source = "throughput"
            elif median:
                eta_s = remaining_steps * median
                eta_source = "step_time"
        summary_duration = _num((self.summary or {}).get("duration_s")) if isinstance(self.summary, dict) else None
        first_time = _num(self.series.records[0].get("time")) if self.series.records else None
        last_time = _num(self.series.records[-1].get("time")) if self.series.records else None
        terminal_times = [_num(terminal.get("time")) for terminal in terminals.values()]
        terminal_times = [value for value in terminal_times if value is not None]
        end_time = max(terminal_times) if terminal_times else last_time
        run_meta = self._manifest.get("run") if isinstance(self._manifest.get("run"), dict) else {}
        process_started = _num(run_meta.get("process_started_at"))
        if completed_now and process_started is not None and end_time is not None:
            elapsed_s = max(0.0, end_time - process_started)
        elif completed_now and summary_duration is not None:
            elapsed_s = summary_duration
        elif completed_now and first_time is not None and last_time is not None:
            elapsed_s = max(0.0, last_time - first_time)
        elif not completed_now and process_started is not None:
            elapsed_s = max(0.0, now - process_started)
        elif self._started_at is not None:
            elapsed_s = max(0.0, now - self._started_at)
        elif first_time is not None and last_time is not None:
            elapsed_s = max(0.0, last_time - first_time)
        else:
            elapsed_s = None
        total_s = elapsed_s + eta_s if elapsed_s is not None and eta_s is not None else None
        if completed_now:
            completion_at = end_time or (
                process_started + elapsed_s if process_started is not None and elapsed_s is not None else now
            )
        else:
            completion_at = now + eta_s if eta_s is not None else None
        loss = self.series.latest_value("loss/total")
        device_used = self.series.latest_value("mem/device_used_gib")
        device_total = self.series.latest_value("mem/device_total_gib")
        last_samples = self.series.latest_value("data/samples")
        last_throughput = self.series.latest_value("perf/samples_per_s")
        last_mfu = self.series.latest_value("perf/mfu")
        if last.get("complete") is False:
            # Rank 0 can only aggregate the ranks whose files it can read.  On node-local metrics that is one
            # node; the mirror has all of them, so the headline batch/throughput/MFU use the global window.
            monitor_throughput, monitor_samples, _ = self.throughput.last_total(self.world)
            if monitor_samples is not None:
                last_samples = monitor_samples
            if monitor_throughput is not None:
                last_throughput = monitor_throughput
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
            "state": run_state,
            "age_s": self.age_s(now),
            "stale": self.age_s(now) > 3 * self.stale_seconds,
            "updated_at": self.updated_at(),
            "started_at": self._started_at,
            "world": self.world,
            "live_ranks": coverage["live_ranks"],
            "terminal_ranks": coverage["terminal_ranks"],
            "coverage": coverage,
            "step": step,
            "max_steps": max_steps,
            "progress": (step / max_steps) if max_steps else None,
            "eta_s": eta_s,
            "eta_source": eta_source,
            "elapsed_s": elapsed_s,
            "total_s": total_s,
            "completion_at": completion_at,
            "recent_throughput": recent_throughput,
            "recent_batch": recent_batch_median,
            "epoch": self.series.latest_value("data/epoch"),
            "last": {
                "step": step,
                "time": _num(last.get("time")),
                "loss": loss,
                "samples": last_samples,
                "vision": self.series.latest_value("loss/vision"),
                "action": self.series.latest_value("loss/action"),
                "lr": self.series.latest_value("optim/lr"),
                "grad_norm": self.series.latest_value("optim/grad_norm"),
                "step_s": step_s,
                "samples_per_s": last_throughput,
                "mfu": last_mfu,
                "device_used_gib": device_used,
                "device_total_gib": device_total,
                "hbm_fraction": (device_used / device_total) if device_used and device_total else None,
                "skew": self.series.latest_value("perf/skew"),
                "complete": last.get("complete"),
            },
            "issues": issues,
            "warnings": self._errors
            + [f"{reader.path}: {reader.error}" for reader in (self.manifest, self.config) if reader and reader.error],
            "hosts": coverage["hosts"],
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

    @staticmethod
    def _control_document(reader: JsonFile) -> Optional[Dict[str, Any]]:
        """Read on demand: a request must be visible the moment the dashboard writes it, and a withdrawn
        one must disappear even though ``JsonFile`` keeps its last good value."""
        value = reader.read()
        if not reader.path.is_file() or not isinstance(value, dict):
            return None
        return dict(value)

    def control_facts(self) -> Dict[str, Any]:
        """The file side of the control channel.  Capabilities are decided by the store, not here."""
        request = self._control_document(self.control_request_file)
        status = self._control_document(self.control_status_file)
        if request is not None:
            request["age_s"] = self.control_request_file.age_s
        if status is not None:
            status["age_s"] = self.control_status_file.age_s
        age = status.get("age_s") if status else None
        return {
            "channel": {
                "present": status is not None,
                "fresh": bool(status is not None and age is not None and age <= STATUS_FRESH_SECONDS),
                "run_started_at": self._started_at,
            },
            "request": request,
            "status": status,
            "checkpoint": latest_generation(checkpoint_root(self._manifest, self.ref.path)),
            "launcher": self.latest_launchers(),
        }

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

    def coverage(self, heartbeats=None, terminals=None) -> Dict[str, Any]:
        """Per-host rank liveness, so a six-node run reports 48 ranks instead of rank 0's eight.

        ``host_ranks`` combines the manifest topology with heartbeat hostnames: a mirrored run may have a
        heartbeat before its manifest is complete, and a node with no heartbeat at all must still appear with
        ``live=0`` rather than silently disappearing from the node count.
        """
        heartbeats = self.latest_heartbeats() if heartbeats is None else heartbeats
        terminals = self.latest_terminals() if terminals is None else terminals
        host_ranks = self._host_rank_map()
        system_hosts = self.system.snapshot().get("hosts", {})
        derived_hosts: Dict[str, bool] = {}
        for row in derive_topology(self._manifest, self.world):
            host = row.get("hostname")
            if host:
                derived_hosts[str(host)] = bool(row.get("derived"))
        rows: List[Dict[str, Any]] = []
        seen = set()
        for host, ranks in sorted(host_ranks.items()):
            ranks = sorted({int(rank) for rank in ranks if rank is not None})
            ages: List[float] = []
            reporting = 0
            for rank in ranks:
                heartbeat = heartbeats.get(rank)
                if not isinstance(heartbeat, dict):
                    continue
                reporting += 1
                age = _num(heartbeat.get("_file_age_s"))
                if age is not None:
                    ages.append(age)
            live = sum(1 for age in ages if age <= self.stale_seconds)
            system_entry = system_hosts.get(host) or {}
            rows.append(
                {
                    "hostname": host,
                    "ranks": ranks,
                    "reporting": reporting,
                    "live": live,
                    "missing": len(ranks) - reporting,
                    "stale": reporting - live,
                    "newest_age_s": min(ages) if ages else None,
                    "oldest_age_s": max(ages) if ages else None,
                    "system_age_s": system_entry.get("age_s"),
                    "derived": derived_hosts.get(host, False),
                }
            )
            seen.add(host)
        for host, system_entry in sorted(system_hosts.items()):
            if host in seen:
                continue
            rows.append(
                {
                    "hostname": host,
                    "ranks": [],
                    "reporting": 0,
                    "live": 0,
                    "missing": 0,
                    "stale": 0,
                    "newest_age_s": None,
                    "oldest_age_s": None,
                    "system_age_s": system_entry.get("age_s"),
                    "derived": True,
                }
            )
        expected = len(derived_hosts) or len(rows)
        return {
            "world": self.world,
            "live_ranks": sum(row["live"] for row in rows),
            "reporting_ranks": len(heartbeats),
            "terminal_ranks": len(terminals),
            "nodes_expected": expected,
            "nodes_reporting": sum(1 for row in rows if row["reporting"]),
            "nodes_live": sum(1 for row in rows if row["live"]),
            "nodes_system": sum(1 for row in rows if row["system_age_s"] is not None),
            "hosts": rows,
        }

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

    def step_detail(self, step: int) -> Optional[Dict[str, Any]]:
        record = self.series.record_for_step(int(step))
        if record is None:
            return None
        args = self._manifest.get("args")
        if isinstance(args, list):
            args = _args_to_dict(args)
        if not isinstance(args, dict):
            config = self._manifest.get("config") if isinstance(self._manifest.get("config"), dict) else {}
            args = _args_to_dict(config.get("args"))
        checkpointing = args.get("activation_checkpointing") or args.get("checkpoint_mode") or "full"
        view = self.throughput.step_view(int(step), self._host_rank_map())
        effective = record
        corrected_by_ranks = False
        if record.get("complete") is False:
            ranks = view.get("ranks") or []
            if ranks:
                effective = dict(record)
                effective["data"] = dict(record.get("data") or {})
                effective["perf"] = dict(record.get("perf") or {})
                effective["data"]["samples"] = sum(_num(entry.get("samples")) or 0.0 for entry in ranks)
                effective["data"].pop("partial_samples", None)
                effective["perf"]["step_ms"] = max((_num(entry.get("step_s")) or 0.0) for entry in ranks) * 1e3
                effective["world"] = self.world
                effective["complete"] = len(ranks) >= self.world
                loss = aggregate_losses(
                    {"loss": entry.get("loss"), "data": {"samples": entry["samples"]}} for entry in ranks
                )
                if loss:
                    effective["loss"] = loss
                effective.pop("samples_estimated", None)
                corrected_by_ranks = True
        detail = stage_metrics(effective, self.world, checkpointing)
        detail.update(
            {
                "run_id": self._manifest.get("run_id") or self.ref.display_name,
                "world": self.world,
                "micro_batch_samples": args.get("samples_per_rank"),
                "grad_accum": args.get("grad_accum"),
                "effective_global_batch": self._manifest.get("effective_global_batch"),
                "throughput": view,
                "record": record,
                "aggregate_complete": record.get("complete", True),
                "aggregate_from_ranks": bool(record.get("aggregate_from_ranks")) or corrected_by_ranks,
                "samples_estimated": bool(record.get("samples_estimated")) and not corrected_by_ranks,
                "reporting_world": record.get("reporting_world"),
            }
        )
        return detail

    def ranks_view(self) -> Dict[str, Any]:
        heartbeats = self.latest_heartbeats()
        terminals = self.latest_terminals()
        host_of = self._rank_hosts()
        world = self.world or (max(heartbeats) + 1 if heartbeats else 0)
        rows = []
        for rank in range(world):
            heartbeat = heartbeats.get(rank)
            terminal = terminals.get(rank)
            age = _num(heartbeat.get("_file_age_s")) if heartbeat else None
            alive = bool(age is not None and age <= self.stale_seconds)
            if terminal and terminal.get("completed"):
                status = "completed"
            elif alive:
                status = "live"
            elif heartbeat:
                status = "stale"
            else:
                status = "missing"
            rows.append(
                {
                    "rank": rank,
                    "hostname": (heartbeat.get("hostname") if heartbeat else None) or host_of.get(rank),
                    "local_rank": heartbeat.get("local_rank") if heartbeat else None,
                    "device_index": heartbeat.get("device_index") if heartbeat else None,
                    "phase": heartbeat.get("phase") if heartbeat else None,
                    "step": heartbeat.get("step") if heartbeat else None,
                    "age_s": age,
                    "status": status,
                    "alive": alive,
                    "heartbeat": heartbeat,
                    "terminal": terminal,
                }
            )
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
