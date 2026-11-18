"""Time: step time statistics, throughput, and how a step splits into forward + backward and the optimizer.

All from the median step: one slow step (a stall, a collective that waited) does not move the headline. A step's time
is that of its slowest rank, which sets the pace of all of them.
"""

import math
import statistics
from typing import Any, Dict, List, Sequence, Tuple


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100), linearly interpolated between the closest ranks (numpy's default)."""
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def slowest_rank_steps(record: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """Per timed step, the time of its slowest rank and that rank's time to the end of the backward."""
    steps, forward_backward = [], []
    for per_rank in zip(*(zip(r["step_s"], r["forward_backward_s"]) for r in record["ranks"])):
        step, until_backward = max(per_rank)
        steps.append(step)
        forward_backward.append(until_backward)
    return steps, forward_backward


def time_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    """Step time, throughput and the step's split of one completed run (``runner.measure`` record)."""
    steps, forward_backward = slowest_rank_steps(record)
    median = statistics.median(steps)
    tokens = sum(r["tokens"] for r in record["ranks"])
    samples = sum(r["samples"] for r in record["ranks"])
    devices = len(record["ranks"])
    forward_backward_median = statistics.median(forward_backward)
    return {
        "step_time_s": {"median": median, "p10": percentile(steps, 10), "p90": percentile(steps, 90)},
        "tokens_per_step": tokens,
        "tokens_per_s": tokens / median,
        "tokens_per_s_per_device": tokens / median / devices,
        "samples_per_s": samples / median,
        # shares of the median step; the optimizer's is the rest
        "forward_backward_s": forward_backward_median,
        "forward_backward_share": forward_backward_median / median,
        "optimizer_share": 1 - forward_backward_median / median,
    }


def loss_finite(record: Dict[str, Any]) -> bool:
    return all(math.isfinite(loss) for r in record["ranks"] for loss in r["losses"])
