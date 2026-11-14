"""In-memory time series derived from aggregated step records.

The browser receives small, decimated arrays; the ECharts ``sampling: 'lttb'`` option handles visual
smoothing.  Decimation here is a cheap bucket min/max so a spike is never dropped between two sampled
points, which a plain stride would do.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Same accounting as ``examples/worldmodel/cosmos3_harness/training_metrics.py``.  Duplicated here so the
#: monitor does not depend on which worktree's metrics module is importable.
ASCEND_910B2_CUBE_PEAK_TFLOPS = 353.9
FWD_TFLOP_PER_SAMPLE = 20.34
VAE_TFLOP_PER_SAMPLE = 33.0


def compute_mfu(step_ms: float, samples: float, world: int, include_vae: bool = False) -> Optional[float]:
    if not step_ms or step_ms <= 0 or world <= 0 or samples <= 0:
        return None
    flops = samples * FWD_TFLOP_PER_SAMPLE * 1e12 * 4.0
    if include_vae:
        flops += samples * VAE_TFLOP_PER_SAMPLE * 1e12
    achieved_tflops = flops / (step_ms / 1e3) / 1e12
    return achieved_tflops / (world * ASCEND_910B2_CUBE_PEAK_TFLOPS)


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _get(record: Dict[str, Any], *path: str) -> Any:
    cursor: Any = record
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


class Series:
    __slots__ = ("x", "y", "unit", "label")

    def __init__(self, unit: str = "", label: str = "") -> None:
        self.x: List[float] = []
        self.y: List[Optional[float]] = []
        self.unit = unit
        self.label = label

    def append(self, x: float, y: Optional[float]) -> None:
        self.x.append(x)
        self.y.append(y)

    def __len__(self) -> int:
        return len(self.x)


#: key -> (unit, label).  Keeping the catalogue here makes the API predictable and the frontend's key list
#: explicit instead of every caller inventing names.
METRICS: Dict[str, Tuple[str, str]] = {
    "loss/total": ("loss", "Loss"),
    "loss/vision": ("loss", "Vision loss"),
    "loss/action": ("loss", "Action loss"),
    "optim/lr": ("lr", "Learning rate"),
    "optim/grad_norm": ("norm", "Grad norm"),
    "perf/step_s": ("s", "Step time"),
    "perf/forward_s": ("s", "Forward"),
    "perf/backward_s": ("s", "Backward"),
    "perf/optim_s": ("s", "Optimizer"),
    "data/wait_s": ("s", "Data wait"),
    "data/vae_s": ("s", "VAE encode"),
    "data/pack_s": ("s", "Pack total"),
    "data/samples": ("count", "Samples"),
    "data/ar_tokens": ("count", "AR tokens"),
    "data/dm_tokens": ("count", "DM tokens"),
    "data/epoch": ("epoch", "Epoch"),
    "perf/samples_per_s": ("samples/s", "Throughput"),
    "perf/mfu": ("%", "MFU"),
    "perf/skew": ("x", "Rank skew"),
    "mem/device_used_gib": ("GiB", "Device used"),
    "mem/peak_reserved_gib": ("GiB", "Peak reserved"),
    "mem/allocated_gib": ("GiB", "Allocated"),
    "mem/device_total_gib": ("GiB", "Device total"),
}


class SeriesStore:
    """All scalar history of one run, plus the raw aggregated records for the Records table."""

    def __init__(self, max_points: int = 200000) -> None:
        self.series: Dict[str, Series] = {key: Series(unit, label) for key, (unit, label) in METRICS.items()}
        self.records: List[Dict[str, Any]] = []
        self.max_points = int(max_points)
        self._last_step: Optional[int] = None
        self._world = 0

    def set_world(self, world: int) -> None:
        self._world = max(0, int(world or 0))

    def clear(self) -> None:
        for series in self.series.values():
            series.x.clear()
            series.y.clear()
        self.records.clear()
        self._last_step = None

    def append(self, record: Dict[str, Any]) -> bool:
        """Add one aggregated record.  Returns False for a duplicate or an out-of-order step.

        Duplicate steps can appear when a mirror rewrites a file or a resumed job replays its last step.
        Keeping the first occurrence makes the chart stable and avoids double-counting samples.
        """
        if record.get("type") not in (None, "record"):
            return False
        try:
            step = int(record.get("step"))
        except (TypeError, ValueError):
            return False
        if self._last_step is not None and step <= self._last_step:
            return False
        self._last_step = step
        self.records.append(record)
        if len(self.records) > self.max_points:
            del self.records[: len(self.records) - self.max_points]
        x = float(step)
        world = _number(record.get("world")) or self._world
        if world:
            self._world = int(world)
        step_ms = _number(_get(record, "perf", "step_ms"))
        samples = _number(_get(record, "data", "samples"))
        if samples is None:
            samples = _number(_get(record, "data", "partial_samples"))
        complete = record.get("complete", True) is not False
        values: Dict[str, Optional[float]] = {
            "loss/total": _number(_get(record, "loss", "total")),
            "loss/vision": _number(_get(record, "loss", "vision")),
            "loss/action": _number(_get(record, "loss", "action")),
            "optim/lr": _number(_get(record, "optim", "lr")),
            "optim/grad_norm": _number(_get(record, "optim", "grad_norm")),
            "perf/step_s": step_ms / 1e3 if step_ms is not None else None,
            "perf/forward_s": _ms(_get(record, "perf", "forward_ms") or _get(record, "perf", "fwd_ms")),
            "perf/backward_s": _ms(_get(record, "perf", "backward_ms") or _get(record, "perf", "bwd_ms")),
            "perf/optim_s": _ms(_get(record, "perf", "optim_ms")),
            "data/wait_s": _ms(_get(record, "data", "wait_ms")),
            "data/vae_s": _ms(_get(record, "data", "vae_ms")),
            "data/pack_s": _ms(_get(record, "data", "pack_ms")),
            "data/samples": samples,
            "data/ar_tokens": _number(_get(record, "data", "ar_tokens")),
            "data/dm_tokens": _number(_get(record, "data", "dm_tokens")),
            "data/epoch": _number(_get(record, "data", "epoch")),
            "perf/skew": _number(_get(record, "perf", "skew")),
            "mem/device_used_gib": _number(_get(record, "mem", "device_used_gib")),
            "mem/peak_reserved_gib": _number(_get(record, "mem", "peak_reserved_gib")),
            "mem/allocated_gib": _number(_get(record, "mem", "allocated_gib")),
            "mem/device_total_gib": _number(_get(record, "mem", "device_total_gib")),
        }
        throughput = (
            samples / (step_ms / 1e3)
            if complete and samples is not None and step_ms is not None and step_ms > 0
            else None
        )
        values["perf/samples_per_s"] = throughput
        values["perf/mfu"] = (
            compute_mfu(step_ms, samples, int(world), include_vae=False)
            if complete and step_ms is not None and samples is not None
            else None
        )
        for key, value in values.items():
            self.series[key].append(x, value)
        return True

    def last(self) -> Optional[Dict[str, Any]]:
        return self.records[-1] if self.records else None

    def latest_value(self, key: str) -> Optional[float]:
        series = self.series.get(key)
        if series is None:
            return None
        for value in reversed(series.y):
            if value is not None:
                return value
        return None

    def query(
        self,
        keys: Iterable[str],
        *,
        x_axis: str = "step",
        max_points: int = 5000,
        from_step: Optional[float] = None,
        to_step: Optional[float] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in keys:
            series = self.series.get(key)
            if series is None:
                continue
            points = _window(series, from_step, to_step)
            xs, ys = _decimate(points, max_points)
            if x_axis == "time":
                xs = _time_axis(self.records, from_step, to_step, len(xs))
            result[key] = {
                "x": xs,
                "y": ys,
                "unit": series.unit,
                "label": series.label,
                "total": len(points),
            }
        return result

    def record_for_step(self, step: int) -> Optional[Dict[str, Any]]:
        for record in reversed(self.records):
            try:
                if int(record.get("step")) == int(step):
                    return record
            except (TypeError, ValueError):
                continue
        return None

    def records_page(self, offset: int = 0, limit: int = 100, order: str = "asc") -> Dict[str, Any]:
        total = len(self.records)
        rows = list(self.records)
        if order == "desc":
            rows = rows[::-1]
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 2000))
        page = rows[offset : offset + limit]
        return {"total": total, "offset": offset, "limit": limit, "rows": page}


def _ms(value: Any) -> Optional[float]:
    number = _number(value)
    return number / 1e3 if number is not None else None


def _window(series: Series, from_step: Optional[float], to_step: Optional[float]):
    if from_step is None and to_step is None:
        return list(zip(series.x, series.y))
    low = -math.inf if from_step is None else from_step
    high = math.inf if to_step is None else to_step
    return [(x, y) for x, y in zip(series.x, series.y) if low <= x <= high]


def _decimate(
    points: List[Tuple[float, Optional[float]]], max_points: int
) -> Tuple[List[float], List[Optional[float]]]:
    """Bucket min/max decimation: preserve extremes, keep the first/last point.

    A plain stride hides a one-point loss spike, exactly the event the dashboard exists to show.  For each
    bucket this keeps the lowest and highest finite y (in original order) and a NaN/None marker if the bucket
    contained one, so gaps stay visible.
    """
    max_points = max(2, int(max_points))
    if len(points) <= max_points:
        return [x for x, _ in points], [y for _, y in points]
    bucket = max(1, len(points) // max_points)
    out: List[Tuple[float, Optional[float]]] = []
    for start in range(0, len(points), bucket):
        chunk = points[start : start + bucket]
        if not chunk:
            continue
        finite = [(x, y) for x, y in chunk if y is not None]
        if not finite:
            out.append((chunk[0][0], None))
            continue
        low = min(finite, key=lambda item: item[1])
        high = max(finite, key=lambda item: item[1])
        if chunk[0][1] is None:
            # Preserve the fact that this bucket contained a gap, without inventing one when it did not.
            out.append((chunk[0][0], None))
        for item in sorted({low, high}, key=lambda pair: pair[0]):
            out.append(item)
    return [x for x, _ in out], [y for _, y in out]


def _time_axis(records: List[Dict[str, Any]], from_step, to_step, count: int) -> List[float]:
    """Return elapsed seconds from the first record for the decimated x values.

    The decimated x values are step numbers; ``count`` lets us return a matching time axis by sampling the
    record timestamps.  When the lengths differ (records were pruned), fall back to the record index.
    """
    rows = records
    if from_step is not None or to_step is not None:
        low = -math.inf if from_step is None else from_step
        high = math.inf if to_step is None else to_step
        rows = [row for row in records if low <= _number(row.get("step")) <= high]
    if not rows:
        return []
    base = _number(rows[0].get("time")) or 0.0
    times = [(_number(row.get("time")) or base) - base for row in rows]
    if len(times) == count:
        return times
    if count <= 1:
        return [0.0]
    step = max(1, len(times) // count)
    sampled = [times[min(index * step, len(times) - 1)] for index in range(count)]
    return sampled


#: Stage accounting for the step-detail view.  The total step is 4x the forward FLOPs when activation
#: checkpointing is on (1 forward + 1 recompute + 2 backward), 3x when it is off.  Allocating the recompute to
#: the backward stage lets a stage MFU be meaningful instead of pretending the forward did all the work.
def stage_metrics(record: Dict[str, Any], world: int, activation_checkpointing: str = "full") -> Dict[str, Any]:
    samples = _number(_get(record, "data", "samples"))
    if samples is None:
        samples = _number(_get(record, "data", "partial_samples"))
    samples = samples or 0.0
    step_ms = _number(_get(record, "perf", "step_ms"))
    step_s = (step_ms / 1e3) if step_ms else None
    checkpointing = str(activation_checkpointing or "full").lower()
    backward_factor = 3.0 if checkpointing in ("full", "selective") else 2.0

    def stage(name, ms, flops, source, note=""):
        seconds = (ms / 1e3) if isinstance(ms, (int, float)) and ms is not None else None
        share = (seconds / step_s) if seconds is not None and step_s else None
        mfu = None
        if seconds and seconds > 0 and flops and world > 0:
            mfu = flops / seconds / (world * ASCEND_910B2_CUBE_PEAK_TFLOPS * 1e12)
        return {
            "name": name,
            "seconds": seconds,
            "share": share,
            "flops": flops,
            "mfu": mfu,
            "source": source,
            "note": note,
        }

    wait_ms = _number(_get(record, "data", "wait_ms"))
    pack_ms = _number(_get(record, "data", "pack_ms"))
    vae_ms = _number(_get(record, "data", "vae_ms"))
    forward_ms = _number(_get(record, "perf", "forward_ms")) or _number(_get(record, "perf", "fwd_ms"))
    backward_ms = _number(_get(record, "perf", "backward_ms")) or _number(_get(record, "perf", "bwd_ms"))
    optim_ms = _number(_get(record, "perf", "optim_ms"))
    prep_ms = None
    if pack_ms is not None:
        prep_ms = max(0.0, pack_ms - (vae_ms or 0.0))
    forward_flops = samples * FWD_TFLOP_PER_SAMPLE * 1e12
    vae_flops = samples * VAE_TFLOP_PER_SAMPLE * 1e12
    stages = [
        stage("data wait", wait_ms, None, "host", "loader / prefetch"),
        stage("data prep", prep_ms, None, "device", "pack, h2d, tokenize"),
        stage("VAE", vae_ms, vae_flops if vae_ms else None, "device", "frozen encoder"),
        stage("forward", forward_ms, forward_flops, "device", "1x fwd FLOPs"),
        stage("backward", backward_ms, forward_flops * backward_factor, "device", f"{backward_factor:g}x fwd FLOPs"),
        stage("optimizer", optim_ms, None, "device", "elementwise update"),
    ]
    known = sum(item["seconds"] for item in stages if item["seconds"] is not None)
    other_s = max(0.0, (step_s - known)) if step_s is not None else None
    stages.append(stage("other", (other_s * 1e3) if other_s is not None else None, None, "host", "sync / logging"))
    total_mfu = compute_mfu(step_ms or 0.0, samples, int(world or 0), include_vae=False)
    return {
        "iteration": _number(record.get("step")),
        "epoch": _number(_get(record, "data", "epoch")),
        "batch_samples": samples,
        "micro_batches": _number(_get(record, "data", "microsteps")),
        "step_seconds": step_s,
        "total_mfu": total_mfu,
        "activation_checkpointing": checkpointing,
        "stages": stages,
        "samples_per_second": (samples / step_s) if step_s else None,
        "loss": _get(record, "loss"),
        "optim": _get(record, "optim"),
        "memory": _get(record, "mem"),
        "complete": record.get("complete"),
    }
