"""Multi-device: the communication shares of a step, and data-parallel scaling efficiency.

Communication is the share of the profiled steps' device time in which communication kernels run; exposed, the share
in which they run while no other kernel does: the time the device waits for communication, which data-parallel scaling
loses. Both are the largest over ranks.

Scaling is weak: every device keeps its micro-batch and sequence length, so N devices train N times the tokens of one
per step. Perfect scaling keeps the step time; the efficiency at N devices is

    (tokens/s at N ÷ N) ÷ (tokens/s at the base ÷ base)

where the base is the smallest device count measured (one device, unless the model only fits sharded over more).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..runner.timeline import Span, union


def communication_shares(spans: Dict[str, List[Span]]) -> Dict[str, float]:
    """Shares of one rank's profiled steps (``runner.timeline.spans_from_trace``)."""
    steps = spans["steps"]
    total = sum(end - start for start, end in steps)
    communication = _clip(spans["communication"], steps)
    exposed = _subtract(communication, union(spans["compute"]))
    return {"communication": _length(communication) / total, "exposed": _length(exposed) / total}


def communication_metrics(record: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """The largest shares over the ranks of a completed run; None when it recorded no timeline (not profiled, or a
    device whose profiler is not read)."""
    shares = [communication_shares(r["timeline"]) for r in record["ranks"] if r["timeline"] is not None]
    if not shares:
        return None
    return {key: max(s[key] for s in shares) for key in ("communication", "exposed")}


def scaling_efficiency(per_device: Sequence[Tuple[int, float]]) -> List[float]:
    """For ``(devices, tokens/s per device)`` points in increasing device counts, each one's efficiency against the
    first."""
    base = per_device[0][1]
    return [throughput / base for _, throughput in per_device]


def _clip(spans: Sequence[Span], windows: Sequence[Span]) -> List[Span]:
    return union([(max(s, ws), min(e, we)) for s, e in spans for ws, we in windows if min(e, we) > max(s, ws)])


def _subtract(spans: Sequence[Span], holes: Sequence[Span]) -> List[Span]:
    """``spans`` minus ``holes``, both sorted and disjoint."""
    result = []
    for start, end in spans:
        for hole_start, hole_end in holes:
            if hole_end <= start or hole_start >= end:
                continue
            if hole_start > start:
                result.append((start, hole_start))
            start = max(start, hole_end)
            if start >= end:
                break
        if start < end:
            result.append((start, end))
    return result


def _length(spans: Sequence[Span]) -> float:
    return sum(end - start for start, end in spans)
