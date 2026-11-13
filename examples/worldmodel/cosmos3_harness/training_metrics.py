r"""Structured, low-overhead observability for training loops.

A training run needs three things it can answer afterwards: *what code and configuration produced these numbers*
(the run manifest), *what happened at every step* (records), and *what went wrong without crashing* (events). This
module owns the record format, the sinks and the cross-rank aggregation; the training script only fills in the
sections it can measure.

Design contract, in order of importance:

1. **The JSONL file is the source of truth.** Weights & Biases and TensorBoard are optional adapters that consume
   the same flattened record. A cluster without network access still gets a complete, inspectable history.
2. **Synchronization belongs to the caller.** Every value in a record is either a host-side number or a scalar the
   caller already read. Section timings use accelerator events recorded lazily; the caller synchronizes measured
   optimizer windows before reading them. Host dispatch timings and fallback sources are reported separately.
3. **The per-step path is a buffered append.** Files are line-buffered and never ``fsync``-ed; a crash loses at
   most the last few records, not the run.
4. **Aggregation is explicit about reduction.** Counts sum, with callers excluding CP copies; step times take
   the maximum (the straggler). Losses take the rank mean. Callers must normalize to their declared objective:
   ``local_sum * dp / global_count`` yields a global mean over samples for DROID or supervised elements for the
   generic replay objective. Equal-sized SFT packs already have equal sample weight; CP copies repeat that loss.

Typical use::

    metrics = TrainingMetrics(out_dir="run/metrics", rank=rank, world_size=world_size, log_every=10,
                              aggregate_observers=[ConsoleObserver()])
    metrics.start(manifest)
    for step in ...:
        metrics.step(step, data={"samples": 8}, perf={"step_ms": step_ms}, mem=device_memory_gib())
    metrics.finish({"steps": step})

Rank 0 reads the other ranks' JSONL files at log points and feeds the aggregated record to the console, W&B and
TensorBoard sinks. Files are read incrementally with a byte offset, one log interval of lag is tolerated, and a
missing file (a rank that has not started yet) is skipped rather than fatal.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SCHEMA_VERSION",
    "MetricsObserver",
    "CompositeObserver",
    "ConsoleObserver",
    "JsonlObserver",
    "TensorboardObserver",
    "WandbObserver",
    "RankFileReader",
    "MetricsAggregator",
    "AnomalyDetector",
    "Stopwatch",
    "TrainingMetrics",
    "aggregate_records",
    "device_memory_gib",
    "flatten",
    "make_event",
    "make_record",
]

SCHEMA_VERSION = 1
RECORD = "record"
EVENT = "event"

#: How each leaf is reduced across ranks. Keys use the flattened ``section/key`` names. Anything not listed keeps
#: rank 0's value (``first``).
REDUCTIONS: Dict[str, str] = {
    # Counts add up: eight ranks with 8 samples each packed 64 samples this step.
    "data/samples": "sum",
    "data/supervised_tokens": "sum",
    "data/ar_tokens": "sum",
    "data/dm_tokens": "sum",
    "data/pad_tokens": "sum",
    "data/real_tokens": "sum",
    "data/skipped_samples": "sum",
    # Durations report the slowest rank: that is the one the step waited for.
    "data/wait_ms": "max",
    "data/pack_ms": "max",
    "data/vae_ms": "max",
    "perf/step_ms": "max",
    "perf/forward_ms": "max",
    "perf/backward_ms": "max",
    "perf/optim_ms": "max",
    "perf/host_dispatch_ms": "max",
    "perf/step_dispatch_ms": "max",
    "mem/allocated_gib": "max",
    "mem/peak_gib": "max",
    "ckpt/save_ms": "max",
    "ckpt/async_pending": "max",
    # Callers normalize losses for their declared sample/element objective; see the module docstring.
    "loss/total": "mean",
    "loss/vision": "mean",
    "loss/action": "mean",
    "optim/grad_norm": "mean",
    "optim/lr": "first",
    "optim/skipped": "sum",
    "data/epoch": "first",
}


# ---------------------------------------------------------------------------------------------------------------
# Records and events
# ---------------------------------------------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """Convert a value to something ``json`` accepts.

    A scalar tensor is converted with ``.item()``, which synchronizes the device. Callers that care about the hot
    path must read their scalars once, at log steps, and pass the resulting Python numbers; see :class:`Stopwatch`
    for section timings that avoid this entirely.
    """
    if isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def make_record(step: int, rank: int, time_s: Optional[float] = None, **sections: Optional[Mapping[str, Any]]) -> Dict:
    """Build one step record, dropping sections and leaves that are ``None``.

    Args:
        step (int): Optimizer step this record belongs to.
        rank (int): Rank that produced it.
        time_s (float, optional): Wall-clock timestamp; defaults to now.
        **sections: Named sections such as ``data``, ``loss``, ``perf``, ``mem``, ``ckpt``. A ``None`` section or
            leaf is omitted so a partial record does not lie with zeros.

    Returns:
        dict: ``{"type": "record", "schema", "step", "rank", "time", ...sections}``.
    """
    record: Dict[str, Any] = {
        "type": RECORD,
        "schema": SCHEMA_VERSION,
        "step": int(step),
        "rank": int(rank),
        "time": time.time() if time_s is None else float(time_s),
    }
    for name, values in sections.items():
        if values is None:
            continue
        kept = {key: _plain(value) for key, value in values.items() if value is not None}
        if kept:
            record[name] = kept
    return record


def make_event(
    step: int,
    kind: str,
    message: str,
    severity: str = "warn",
    rank: int = 0,
    time_s: Optional[float] = None,
    **values: Any,
) -> Dict:
    """Build one event: a condition worth surfacing that did not stop training."""
    return {
        "type": EVENT,
        "schema": SCHEMA_VERSION,
        "step": int(step),
        "rank": int(rank),
        "time": time.time() if time_s is None else float(time_s),
        "severity": severity,
        "kind": kind,
        "message": message,
        "values": {key: _plain(value) for key, value in values.items() if value is not None},
    }


def flatten(mapping: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested sections to ``section/key`` names, the form the metric sinks log."""
    flat: Dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, prefix=f"{name}/"))
        else:
            flat[name] = value
    return flat


def _unflatten(flat: Mapping[str, Any]) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split("/")
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


def _reduce(values: Sequence[Any], policy: str) -> Any:
    if policy == "sum":
        return sum(values)
    if policy == "max":
        return max(values)
    if policy == "min":
        return min(values)
    if policy == "mean":
        return statistics.fmean(values)
    if policy == "last":
        return values[-1]
    return values[0]  # "first"


def aggregate_records(records: Iterable[Optional[Mapping[str, Any]]]) -> Optional[Dict]:
    """Reduce per-rank records of one step into a single record.

    Counts are summed, durations and memory are maximised, losses averaged (see :data:`REDUCTIONS`). Two derived
    fields are added to ``perf`` so a straggler is visible without post-processing: ``step_ms_mean`` and ``skew``
    (``max / mean``).

    Args:
        records: The per-rank records; ``None`` entries are ignored.

    Returns:
        dict or None: The aggregated record, or None when no record was provided.
    """
    present = [record for record in records if record]
    if not present:
        return None

    flats = [flatten(record) for record in present]
    keys = sorted(set().union(*(flat.keys() for flat in flats)))
    reduced: Dict[str, Any] = {}
    for key in keys:
        if key in ("type", "schema", "rank", "step", "time", "ranks", "world"):
            continue
        values = [flat[key] for flat in flats if key in flat]
        if values:
            reduced[key] = _reduce(values, REDUCTIONS.get(key, "first"))

    aggregated = _unflatten(reduced)
    ranks = sorted({int(record.get("rank", 0)) for record in present})
    aggregated.update(
        {
            "type": RECORD,
            "schema": SCHEMA_VERSION,
            "step": max(int(record["step"]) for record in present),
            "rank": 0,
            "ranks": ranks,
            "world": len(ranks),
            "time": max(float(record.get("time", 0.0)) for record in present),
        }
    )

    step_times = [flat["perf/step_ms"] for flat in flats if "perf/step_ms" in flat]
    if len(step_times) > 1:
        mean = statistics.fmean(step_times)
        peak = max(step_times)
        perf = aggregated.setdefault("perf", {})
        perf["step_ms_mean"] = mean
        perf["step_ms_max"] = peak
        perf["skew"] = peak / mean if mean else 1.0
    return aggregated


def device_memory_gib() -> Dict[str, float]:
    """Allocated and peak GiB on the current accelerator, or an empty dict when there is none.

    Both values are host-side allocator counters, so this is free and never synchronizes.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency of the package
        return {}
    for name in ("npu", "cuda"):
        module = getattr(torch, name, None)
        if module is None or not hasattr(module, "memory_allocated"):
            continue
        try:
            return {
                "allocated_gib": module.memory_allocated() / 2**30,
                "peak_gib": module.max_memory_allocated() / 2**30,
            }
        except Exception:  # an accelerator that is present but not initialised
            return {}
    return {}


# ---------------------------------------------------------------------------------------------------------------
# Section timings
# ---------------------------------------------------------------------------------------------------------------


def _events_available() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    if torch.cuda.is_available() and hasattr(torch.cuda, "Event"):
        return True
    return hasattr(torch, "npu") and hasattr(torch.npu, "Event")


class Stopwatch:
    """Measure section durations with accelerator events, read after the step's existing sync.

    ``perf_counter`` around an asynchronous section measures *dispatch*, not device time: a 0.7 s VAE encode can
    look like a 40 ms call because the host returns as soon as the kernels are queued. Recording an event at each
    boundary stamps the device timeline; ``elapsed_time`` is then read at a log step, after the training loop has
    already synchronized to print the loss. No per-section synchronization is added.

    Call :meth:`reset` at the start of every step -- ``enabled`` is true only on log steps, so event recording costs
    nothing on the other 9 of 10 steps.

    Args:
        use_events (bool, optional): Force event or wall-clock timing. Defaults to using events when the current
            torch build exposes ``torch.cuda.Event`` or ``torch.npu.Event`` (falls back to wall-clock on CPU and
            older builds).
    """

    def __init__(self, use_events: Optional[bool] = None) -> None:
        self._use_events = _events_available() if use_events is None else bool(use_events)
        self._enabled = False
        self._spans: Dict[str, List[Tuple[float, float, Any, Any]]] = {}
        self.timing_sources: Dict[str, str] = {}

    def reset(self, enabled: bool = True) -> None:
        """Start a new step; only an enabled step records anything."""
        self._spans.clear()
        self.timing_sources.clear()
        self._enabled = bool(enabled)

    def _record_event(self):
        if not self._use_events:
            return None
        try:
            import torch

            if torch.cuda.is_available():
                event = torch.cuda.Event(enable_timing=True)
            else:
                event = torch.npu.Event(enable_timing=True)
            event.record()
            return event
        except Exception:
            self._use_events = False
            return None

    @contextlib.contextmanager
    def section(self, name: str):
        """Time one section. A no-op on steps where the stopwatch is disabled."""
        if not self._enabled:
            yield
            return
        start = time.perf_counter()
        start_event = self._record_event()
        try:
            yield
        finally:
            end = time.perf_counter()
            end_event = self._record_event()
            self._spans.setdefault(name, []).append((start, end, start_event, end_event))

    def read_ms(self) -> Dict[str, float]:
        """Return ``{section: milliseconds}``.

        Call this only after a point where the device has been synchronized for another reason (``loss.item()`` at
        the log step). Repeated sections are summed across the accumulation window. An unavailable event pair
        falls back to host dispatch time; ``timing_sources`` labels that fallback explicitly.
        """
        timings: Dict[str, float] = {}
        for name, spans in self._spans.items():
            total = 0.0
            sources = set()
            for start, end, start_event, end_event in spans:
                value = None
                if start_event is not None and end_event is not None:
                    try:
                        value = float(start_event.elapsed_time(end_event))
                    except Exception:
                        value = None
                sources.add("device_events" if value is not None else "host_dispatch")
                total += value if value is not None else (end - start) * 1e3
            timings[name] = total
            self.timing_sources[name] = next(iter(sources)) if len(sources) == 1 else "mixed"
        return timings

    def read_dispatch_ms(self) -> Dict[str, float]:
        """Return summed host dispatch durations, separately from device event durations."""
        return {name: sum((end - start) * 1e3 for start, end, _, _ in spans) for name, spans in self._spans.items()}


# ---------------------------------------------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------------------------------------------


class MetricsObserver:
    """Base class for metric sinks. Every hook is optional and defaults to a no-op."""

    def on_run_start(self, manifest: Mapping[str, Any]) -> None:
        pass

    def on_step(self, record: Mapping[str, Any]) -> None:
        pass

    def on_event(self, event: Mapping[str, Any]) -> None:
        pass

    def on_run_end(self, summary: Mapping[str, Any]) -> None:
        pass


class CompositeObserver(MetricsObserver):
    """Fan a hook out to several observers."""

    def __init__(self, observers: Sequence[MetricsObserver]) -> None:
        self.observers = list(observers)

    def on_run_start(self, manifest):
        for observer in self.observers:
            observer.on_run_start(manifest)

    def on_step(self, record):
        for observer in self.observers:
            observer.on_step(record)

    def on_event(self, event):
        for observer in self.observers:
            observer.on_event(event)

    def on_run_end(self, summary):
        for observer in self.observers:
            observer.on_run_end(summary)


class JsonlObserver(MetricsObserver):
    """Append every record and event as one JSON line.

    Writes are line-buffered and never ``fsync``-ed: an NFS stall costs at most this step's record, and a crash at
    worst the tail of the file. This file is the source of truth the other sinks can be rebuilt from.
    """

    def __init__(self, path, flush: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", buffering=1, encoding="utf-8")
        self._flush = flush

    def _write(self, payload: Mapping[str, Any]) -> None:
        self._stream.write(json.dumps(payload, default=_json_default) + "\n")
        if self._flush:
            self._stream.flush()

    def on_step(self, record):
        self._write(record)

    def on_event(self, event):
        self._write(event)

    def on_run_end(self, summary):
        if summary:
            self._write({"type": "summary", "schema": SCHEMA_VERSION, "time": time.time(), "summary": dict(summary)})
        self._stream.close()


class ConsoleObserver(MetricsObserver):
    """One compact line per log step plus one line per event; meant for rank 0."""

    def __init__(self, stream=None) -> None:
        self._stream = stream

    def _print(self, text: str) -> None:
        print(text, file=self._stream, flush=True)

    def on_step(self, record):
        step = record.get("step", "?")
        parts: List[str] = []

        loss = record.get("loss", {})
        if "total" in loss:
            text = f"loss {loss['total']:.6f}"
            if "vision" in loss and "action" in loss:
                text += f" (v {loss['vision']:.4f} a {loss['action']:.4f})"
            parts.append(text)

        data = record.get("data", {})
        if "supervised_tokens" in data:
            tokens = data["supervised_tokens"]
            parts.append(f"tok {tokens / 1e3:.1f}k" if tokens >= 1e3 else f"tok {tokens}")
        if "samples" in data:
            parts.append(f"samples {data['samples']}")

        perf = record.get("perf", {})
        if "step_ms" in perf:
            text = f"step {perf['step_ms'] / 1e3:.2f}s"
            sections = []
            for key in ("forward_ms", "backward_ms", "optim_ms"):
                if key in perf:
                    sections.append(f"{key[:-3]} {perf[key] / 1e3:.2f}")
            if sections:
                text += f" ({', '.join(sections)})"
            parts.append(text)
        if "skew" in perf:
            parts.append(f"skew {perf['skew']:.2f}")

        if "wait_ms" in data or "vae_ms" in data:
            sections = []
            if "vae_ms" in data:
                sections.append(f"vae {data['vae_ms'] / 1e3:.2f}")
            if "wait_ms" in data:
                sections.append(f"wait {data['wait_ms'] / 1e3:.2f}")
            parts.append(f"data ({', '.join(sections)})")

        mem = record.get("mem", {})
        if "allocated_gib" in mem:
            parts.append(f"mem {mem['allocated_gib']:.1f} GiB")

        if "ranks" in record and record.get("world", 1) > 1:
            parts.append(f"ranks {record['world']}")

        self._print(f"[step {step}] " + " | ".join(parts) if parts else f"[step {step}]")

    def on_event(self, event):
        values = event.get("values") or {}
        suffix = f" {values}" if values else ""
        self._print(
            f"[event step {event.get('step', '?')}] {event.get('severity', 'warn')} "
            f"{event.get('kind', 'event')}: {event.get('message', '')}{suffix}"
        )


class WandbObserver(MetricsObserver):
    """Log aggregated records to Weights & Biases.

    ``wandb`` is imported lazily, so a container without it pays nothing until this observer is actually built.
    ``mode="offline"`` is the default because training hosts are often isolated; the run directory can be synced
    later with ``wandb sync``.

    Args:
        project (str): W&B project name.
        run_name (str, optional): Run name; defaults to W&B's generated name.
        mode (str, optional): ``"offline"``, ``"online"`` or ``"disabled"``. Defaults to ``"offline"``.
        config (Mapping, optional): Extra config; the run manifest is always merged in.
        tags (Sequence[str], optional): Run tags.
    """

    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        mode: str = "offline",
        config: Optional[Mapping[str, Any]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> None:
        self.project = project
        self.run_name = run_name
        self.mode = mode
        self.config = dict(config or {})
        self.tags = list(tags or [])
        self._wandb = None

    def on_run_start(self, manifest):
        try:
            self._wandb = importlib.import_module("wandb")
        except ImportError as error:
            raise RuntimeError(
                "WandbObserver needs the 'wandb' package: pip install wandb (the JSONL metrics keep working without "
                "it)."
            ) from error
        self._wandb.init(
            project=self.project,
            name=self.run_name,
            mode=self.mode,
            tags=self.tags,
            config={**self.config, "manifest": dict(manifest or {})},
        )

    def on_step(self, record):
        if self._wandb is None:
            return
        payload = {
            key: value
            for key, value in flatten(record).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        self._wandb.log(payload, step=record.get("step"))

    def on_event(self, event):
        if self._wandb is None:
            return
        self._wandb.log({f"events/{event.get('kind', 'event')}": 1}, step=event.get("step"))

    def on_run_end(self, summary):
        if self._wandb is None:
            return
        if summary:
            self._wandb.log(
                {f"summary/{key}": value for key, value in summary.items() if isinstance(value, (int, float))}
            )
        self._wandb.finish()
        self._wandb = None


class TensorboardObserver(MetricsObserver):
    """Log aggregated records to a local TensorBoard event file. Lazy import, same contract as W&B."""

    def __init__(self, log_dir) -> None:
        self.log_dir = str(log_dir)
        self._writer = None

    def on_run_start(self, manifest):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorboardObserver needs torch.utils.tensorboard (pip install tensorboard). The JSONL metrics keep "
                "working without it."
            ) from error
        self._writer = SummaryWriter(log_dir=self.log_dir)
        if manifest:
            self._writer.add_text("run/manifest", f"```json\n{json.dumps(dict(manifest), indent=2)}\n```")

    def on_step(self, record):
        if self._writer is None:
            return
        step = record.get("step", 0)
        for key, value in flatten(record).items():
            if isinstance(value, (int, float, bool)) and not isinstance(value, bool):
                self._writer.add_scalar(key, value, global_step=step)

    def on_event(self, event):
        if self._writer is None:
            return
        self._writer.add_text(
            f"events/{event.get('kind', 'event')}",
            f"[{event.get('severity', 'warn')}] step {event.get('step')}: {event.get('message')}",
            global_step=event.get("step", 0),
        )

    def on_run_end(self, summary):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# ---------------------------------------------------------------------------------------------------------------
# Cross-rank aggregation
# ---------------------------------------------------------------------------------------------------------------


class RankFileReader:
    """Incrementally read one rank's JSONL file.

    Records are indexed by step and events are buffered per step; ``read_new`` consumes only the bytes appended
    since the last call, so rank 0 does not re-parse the whole history every log interval. A partially written last
    line is kept and completed on the next read.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._offset = 0
        self._tail = ""
        self._records: Dict[int, Dict] = {}
        self._events: Dict[int, List[Dict]] = {}

    def read_new(self) -> None:
        try:
            with self.path.open("rb") as stream:
                stream.seek(self._offset)
                chunk = stream.read()
        except FileNotFoundError:
            return
        if not chunk:
            return
        self._offset += len(chunk)
        text = self._tail + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._tail = lines.pop()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = int(payload.get("step", -1))
            if payload.get("type") == EVENT:
                self._events.setdefault(step, []).append(payload)
            elif payload.get("type") == RECORD:
                self._records[step] = payload

    def pop(self, step: int) -> Tuple[Optional[Dict], List[Dict]]:
        """Take the record for ``step`` and every event up to it.

        Records older than ``step`` are dropped: a rank that wrote after the aggregator moved on would otherwise
        accumulate one dict per step for the rest of the run.
        """
        record = self._records.pop(step, None)
        for stale in [key for key in self._records if key < step]:
            self._records.pop(stale)
        events: List[Dict] = []
        for event_step in [key for key in self._events if key <= step]:
            events.extend(self._events.pop(event_step))
        return record, events


class MetricsAggregator:
    """Rank 0's view of every rank: incremental readers plus :func:`aggregate_records`."""

    def __init__(self, directory, world_size: int, rank: int = 0) -> None:
        self.directory = Path(directory)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._readers = {
            index: RankFileReader(self.directory / f"rank-{index:03d}.jsonl") for index in range(self.world_size)
        }

    def collect(self, step: int, own_record: Optional[Mapping[str, Any]] = None) -> Tuple[List[Dict], List[Dict]]:
        """Return the records and events of ``step`` from every rank that has written them."""
        records: List[Dict] = []
        events: List[Dict] = []
        for index, reader in self._readers.items():
            if index == self.rank and own_record is not None:
                records.append(dict(own_record))
                continue
            reader.read_new()
            record, rank_events = reader.pop(step)
            if record is not None:
                records.append(record)
            events.extend(rank_events)
        return records, events


# ---------------------------------------------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------------------------------------------


class AnomalyDetector:
    """Flag the failure modes that degrade a run without stopping it.

    The checks are intentionally cheap and run on the aggregated record at log steps: a non-finite loss, a loss
    spike or collapse against an EMA, a step-time jump against the rolling median, rank skew, dataloader
    starvation, and memory pressure. Each hit becomes an event, so it reaches the console and the dashboard within
    one log interval instead of being discovered later from the loss curve.

    Args:
        warmup (int, optional): Steps before the ratio checks arm. Defaults to 10.
        loss_spike_ratio (float, optional): Loss/EMA above this is a spike. Defaults to 3.0.
        loss_drop_ratio (float, optional): Loss/EMA below this is a collapse. Defaults to 1/3.
        step_time_ratio (float, optional): Step time over this multiple of the rolling median. Defaults to 1.5.
        skew_ratio (float, optional): Slowest/mean step time above this means a straggler. Defaults to 1.2.
        wait_fraction (float, optional): Dataloader wait over this fraction of the step. Defaults to 0.3.
        memory_fraction (float, optional): Allocated memory over this fraction of the device. Defaults to 0.9.
        device_capacity_gib (float, optional): Device HBM in GiB; memory checks are skipped without it.
        window (int, optional): Steps in the rolling step-time median. Defaults to 20.
    """

    def __init__(
        self,
        warmup: int = 10,
        loss_spike_ratio: float = 3.0,
        loss_drop_ratio: float = 1.0 / 3.0,
        step_time_ratio: float = 1.5,
        skew_ratio: float = 1.2,
        wait_fraction: float = 0.3,
        memory_fraction: float = 0.9,
        device_capacity_gib: Optional[float] = None,
        window: int = 20,
    ) -> None:
        self.warmup = int(warmup)
        self.loss_spike_ratio = float(loss_spike_ratio)
        self.loss_drop_ratio = float(loss_drop_ratio)
        self.step_time_ratio = float(step_time_ratio)
        self.skew_ratio = float(skew_ratio)
        self.wait_fraction = float(wait_fraction)
        self.memory_fraction = float(memory_fraction)
        self.device_capacity_gib = device_capacity_gib
        self._window = int(window)
        self._loss_ema: Optional[float] = None
        self._step_times: List[float] = []
        self._seen = 0

    def check(self, record: Mapping[str, Any]) -> List[Dict]:
        """Return the events this record triggers. State advances by one step per call."""
        events: List[Dict] = []
        step = int(record.get("step", 0))
        armed = self._seen >= self.warmup

        loss = record.get("loss", {}).get("total")
        if loss is not None:
            if not math.isfinite(loss):
                events.append(
                    make_event(
                        step,
                        "nonfinite_loss",
                        f"loss is {loss}; the step should be skipped",
                        "error",
                        rank=int(record.get("rank", 0)),
                    )
                )
                self._loss_ema = None
            else:
                if self._loss_ema is None:
                    self._loss_ema = loss
                else:
                    ratio = loss / self._loss_ema if self._loss_ema else 1.0
                    if armed and (ratio > self.loss_spike_ratio or ratio < self.loss_drop_ratio):
                        events.append(
                            make_event(
                                step,
                                "loss_spike" if ratio > 1.0 else "loss_drop",
                                f"loss {loss:.6g} is {ratio:.2f}x the EMA {self._loss_ema:.6g}",
                                "warn",
                                ratio=ratio,
                                loss=loss,
                                ema=self._loss_ema,
                            )
                        )
                    self._loss_ema = 0.9 * self._loss_ema + 0.1 * loss

        perf = record.get("perf", {})
        step_ms = perf.get("step_ms")
        if step_ms:
            median = statistics.median(self._step_times) if len(self._step_times) >= 5 else None
            if median and armed and step_ms > self.step_time_ratio * median:
                events.append(
                    make_event(
                        step,
                        "step_time_jump",
                        f"step {step_ms / 1e3:.2f}s is {step_ms / median:.2f}x the median {median / 1e3:.2f}s",
                        "warn",
                        step_ms=step_ms,
                        median_ms=median,
                    )
                )
            self._step_times = (self._step_times + [step_ms])[-self._window :]

        skew = perf.get("skew")
        if skew is not None and skew > self.skew_ratio:
            events.append(
                make_event(
                    step,
                    "rank_skew",
                    f"slowest rank is {skew:.2f}x the mean step time",
                    "warn",
                    skew=skew,
                )
            )

        data = record.get("data", {})
        wait = data.get("wait_ms")
        if wait is not None and step_ms and wait > self.wait_fraction * step_ms:
            events.append(
                make_event(
                    step,
                    "data_starvation",
                    f"dataloader wait {wait:.0f}ms is {wait / step_ms:.0%} of the {step_ms / 1e3:.2f}s step",
                    "warn",
                    wait_ms=wait,
                    step_ms=step_ms,
                )
            )

        mem = record.get("mem", {})
        allocated = mem.get("allocated_gib")
        if allocated is not None and self.device_capacity_gib:
            fraction = allocated / self.device_capacity_gib
            if fraction > self.memory_fraction:
                events.append(
                    make_event(
                        step,
                        "memory_pressure",
                        f"allocated {allocated:.1f} GiB is {fraction:.0%} of the "
                        f"{self.device_capacity_gib:.1f} GiB device",
                        "warn",
                        allocated_gib=allocated,
                        fraction=fraction,
                    )
                )

        self._seen += 1
        return events


# ---------------------------------------------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------------------------------------------


class TrainingMetrics:
    """The object a training loop talks to.

    It owns the rank-local JSONL sink, the rank-0 aggregation and the optional dashboards. Call :meth:`step` once
    per optimizer step with whatever sections are available; expensive scalars (loss) only on log steps, cheap
    host-side ones (tokens, step time, memory) every step.

    Args:
        out_dir (str or Path, optional): Directory for ``rank-*.jsonl``, ``run.json`` and ``summary.json``. Without
            it, only the observers passed in are used.
        rank (int, optional): This process's rank. Defaults to 0.
        world_size (int, optional): Number of ranks. Defaults to 1.
        log_every (int, optional): Aggregate and report every N steps. Defaults to 10.
        local_observers (Sequence[MetricsObserver], optional): Sinks fed on this rank only.
        aggregate_observers (Sequence[MetricsObserver], optional): Sinks fed on rank 0 with the aggregated record.
        detector (AnomalyDetector, optional): Runs on the aggregated record at log steps.
        write_jsonl (bool, optional): Create the rank-local JSONL sink. Defaults to True; ignored without out_dir.
    """

    def __init__(
        self,
        out_dir=None,
        rank: int = 0,
        world_size: int = 1,
        log_every: int = 10,
        local_observers: Sequence[MetricsObserver] = (),
        aggregate_observers: Sequence[MetricsObserver] = (),
        detector: Optional[AnomalyDetector] = None,
        write_jsonl: bool = True,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.log_every = max(1, int(log_every))
        self.out_dir = Path(out_dir) if out_dir is not None else None

        local = list(local_observers)
        if write_jsonl and self.out_dir is not None:
            local.insert(0, JsonlObserver(self.out_dir / f"rank-{self.rank:03d}.jsonl"))
        self.local_observers = local
        self.aggregate_observers = list(aggregate_observers)
        self.detector = detector

        self._aggregator = None
        if self.out_dir is not None and self.rank == 0 and self.world_size > 1:
            self._aggregator = MetricsAggregator(self.out_dir, self.world_size, rank=self.rank)
        self._pending_events: Dict[int, List[Dict]] = {}

    # -- lifecycle ----------------------------------------------------------------------------------------------

    def start(self, manifest: Optional[Mapping[str, Any]] = None) -> None:
        """Write the run manifest on rank 0 and notify the sinks."""
        if manifest is not None and self.out_dir is not None and self.rank == 0:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "run.json").write_text(
                json.dumps(dict(manifest), indent=2, default=_json_default) + "\n", encoding="utf-8"
            )
        for observer in self.local_observers + self.aggregate_observers:
            observer.on_run_start(manifest or {})

    def finish(self, summary: Optional[Mapping[str, Any]] = None) -> None:
        """Write the summary on rank 0, forward any still-queued events, and close every sink."""
        if summary is not None and self.out_dir is not None and self.rank == 0:
            (self.out_dir / "summary.json").write_text(
                json.dumps(dict(summary), indent=2, default=_json_default) + "\n", encoding="utf-8"
            )
        if self.rank == 0:
            # An event after the last log step has no step left to be forwarded at; report it with the summary.
            for pending_step in sorted(self._pending_events):
                for event in self._pending_events.pop(pending_step):
                    for observer in self.aggregate_observers:
                        observer.on_event(event)
        for observer in self.local_observers + self.aggregate_observers:
            observer.on_run_end(summary or {})

    # -- data ---------------------------------------------------------------------------------------------------

    def event(self, step: int, kind: str, message: str, severity: str = "warn", **values: Any) -> None:
        """Report a rank-local event immediately; rank 0 forwards it at the next log step."""
        event = make_event(step, kind, message, severity, rank=self.rank, **values)
        for observer in self.local_observers:
            observer.on_event(event)
        if self.rank == 0:
            # Other ranks' events reach rank 0 through their JSONL files; only rank 0's own queue is needed.
            self._pending_events.setdefault(int(step), []).append(event)

    def step(self, step: int, **sections: Optional[Mapping[str, Any]]) -> None:
        """Record one optimizer step.

        Args:
            step (int): Optimizer step, 1-based.
            **sections: ``data``, ``loss``, ``optim``, ``perf``, ``mem``, ``ckpt``; ``None`` sections are omitted.
        """
        record = make_record(step, self.rank, **sections)
        for observer in self.local_observers:
            observer.on_step(record)

        if self.rank != 0 or not (int(step) == 1 or int(step) % self.log_every == 0):
            return

        records, events = [dict(record)], [
            event
            for pending_step in sorted(self._pending_events)
            if pending_step <= int(step)
            for event in self._pending_events.pop(pending_step)
        ]
        if self._aggregator is not None:
            gathered, gathered_events = self._aggregator.collect(int(step), own_record=record)
            if gathered:
                records = gathered
            events.extend(gathered_events)

        aggregated = aggregate_records(records) or record
        if self.detector is not None:
            detected = self.detector.check(aggregated)
            for event in detected:
                for observer in self.local_observers:
                    observer.on_event(event)
            events.extend(detected)

        for observer in self.aggregate_observers:
            observer.on_step(aggregated)
            for event in events:
                observer.on_event(event)
