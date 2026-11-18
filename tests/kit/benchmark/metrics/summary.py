"""Summary: every metric of one measurement (a setup run in several repeats), and the metrics that relate several
measurements: a scaling curve and a comparison. Built from ``time``, ``compute``, ``memory``, ``parallel`` and
``checks``; scenarios and the report read these and compute nothing themselves."""

import json
from typing import Any, Dict, List, Optional

from . import checks
from .compute import utilization
from .memory import device_used_at_start, max_over_repeats, memory_metrics
from .parallel import communication_metrics, scaling_efficiency
from .time import loss_finite, time_metrics


def summarize(records: List[Dict[str, Any]], flops: Dict[str, Any], peak: Dict[str, Any]) -> Dict[str, Any]:
    """The metrics of one setup measured ``len(records)`` times, each a completed ``runner.measure`` record, with the
    FLOPs per token (``compute.flops_per_token``) and the device's peak (``compute.Peak`` as a dict).

    The headline is the repeat with the median throughput, with an even count the slower of the two middle ones;
    memory is the maximum over repeats."""
    if not records:
        raise ValueError("a measurement needs at least one run")
    if len({json.dumps(r["setup"], sort_keys=True) for r in records}) != 1:
        raise ValueError("every run of a measurement must measure the same setup")
    repeats = []
    for record in records:
        time = time_metrics(record)
        repeats.append(
            {
                "time": time,
                "utilization": utilization(time["tokens_per_s_per_device"], flops, peak),
                "memory_bytes": memory_metrics(record),
                "communication": communication_metrics(record),
                "loss_finite": loss_finite(record),
                "device_used_at_start": device_used_at_start(record),
            }
        )
    medians = [r["time"]["step_time_s"]["median"] for r in repeats]
    order = sorted(range(len(repeats)), key=lambda i: medians[i])
    headline = order[len(order) // 2]
    summary = {
        "devices": len(records[0]["ranks"]),
        "parameters": records[0]["parameters"],
        "optimizer": records[0]["optimizer"],
        "headline_repeat": headline,
        "repeat_spread": (max(medians) - min(medians)) / min(medians),
        "time": repeats[headline]["time"],
        "utilization": repeats[headline]["utilization"],
        "communication": repeats[headline]["communication"],
        "memory_bytes": max_over_repeats([r["memory_bytes"] for r in repeats]),
        "repeats": repeats,
    }
    summary["flags"] = checks.flags(summary, flops)
    return summary


def scaling_curve(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One point per summary of the same setup on increasing device counts, with the efficiency against the first."""
    efficiency = scaling_efficiency([(s["devices"], s["time"]["tokens_per_s_per_device"]) for s in summaries])
    return [
        {
            "devices": s["devices"],
            "tokens_per_s": s["time"]["tokens_per_s"],
            "tokens_per_s_per_device": s["time"]["tokens_per_s_per_device"],
            "efficiency": e,
            "step_time_s": s["time"]["step_time_s"]["median"],
            "mfu_nominal": s["utilization"]["mfu_nominal"],
            "reserved_peak": s["memory_bytes"].get("reserved_peak"),
            "communication": (s["communication"] or {}).get("communication"),
            "exposed_communication": (s["communication"] or {}).get("exposed"),
            "flags": s["flags"],
        }
        for s, e in zip(summaries, efficiency)
    ]


def comparison(ours: Dict[str, Any], theirs: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Ratios of two summaries of setups that differ in one field; above 1, ``ours`` trains faster / uses more."""
    return {
        "throughput_ratio": ours["time"]["tokens_per_s"] / theirs["time"]["tokens_per_s"],
        "reserved_ratio": _ratio(
            ours["memory_bytes"].get("reserved_peak"), theirs["memory_bytes"].get("reserved_peak")
        ),
    }


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return a / b if a is not None and b else None
