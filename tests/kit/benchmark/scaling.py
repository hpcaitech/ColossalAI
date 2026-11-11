"""Data-parallel scaling efficiency: the same setup on 1, 2, 4, 8 devices.

Weak scaling: every device keeps its micro-batch and sequence length, so N devices train N times the tokens of one
per step. Perfect scaling keeps the step time; the efficiency at N devices is

    (tokens/s at N ÷ N) ÷ (tokens/s at the base ÷ base)

where the base is the smallest device count measured (one device, unless the model only fits sharded over more).

Each point also profiles ``PROFILED_STEPS`` steps after its timed ones (outside the timing) to report how much of a
step the devices spend in communication, and how much of that is exposed, i.e. not hidden behind computation: the
usual reason for an efficiency below 1.
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from . import measure, run
from .spec import BenchmarkSpec

DEFAULT_COUNTS = (1, 2, 4, 8)
PROFILED_STEPS = 3


def run_scaling(
    setup: measure.Setup,
    counts: Sequence[int] = DEFAULT_COUNTS,
    repeats: int = run.DEFAULT_REPEATS,
    spec: Optional[BenchmarkSpec] = None,
    command: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Measure ``setup`` on each device count in ``counts`` and return the report of the largest, carrying the whole
    curve under ``"scaling"``."""
    counts = sorted(set(counts))
    if not counts or counts[0] < 1:
        raise ValueError(f"device counts must be positive, got {counts}")
    setup = replace(setup, profile_steps=setup.profile_steps or PROFILED_STEPS)
    reports = [run.run_benchmark(setup, nproc=n, repeats=repeats, spec=spec, command=command) for n in counts]
    result = reports[-1]
    result["scaling"] = scaling_curve(reports)
    return result


def scaling_curve(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One point per report (``run.run_benchmark`` results of the same setup on increasing device counts), with the
    efficiency against the first."""
    base = reports[0]
    base_per_device = base["throughput"]["tokens_per_s_per_device"]
    points = []
    for r in reports:
        throughput = r["throughput"]
        headline = r["repeats"][r["headline_repeat"]]
        points.append(
            {
                "devices": r["world_size"],
                "tokens_per_s": throughput["tokens_per_s"],
                "tokens_per_s_per_device": throughput["tokens_per_s_per_device"],
                "efficiency": throughput["tokens_per_s_per_device"] / base_per_device,
                "step_time_s": throughput["step_time_s"]["median"],
                "mfu_nominal": throughput["mfu_nominal"],
                "reserved_peak": r["memory_bytes"].get("reserved_peak"),
                "communication": _worst(headline.get("communication"), "communication"),
                "exposed_communication": _worst(headline.get("communication"), "exposed"),
                "flags": r["flags"],
            }
        )
    return points


def _worst(per_rank: Optional[List[Optional[Dict[str, float]]]], key: str) -> Optional[float]:
    # the largest share over ranks
    values = [r[key] for r in per_rank or () if r is not None]
    return max(values) if values else None
