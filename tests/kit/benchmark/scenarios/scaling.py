"""Data-parallel scaling: the same setup on 1, 2, 4, 8 devices, changing only ``Setup.devices``. The efficiency and
communication shares are defined in ``metrics.parallel``.

Each point also profiles ``PROFILED_STEPS`` steps after its timed ones (outside the timing) for the communication
shares: the usual reason for an efficiency below 1.
"""

from dataclasses import replace
from typing import Any, Dict, Optional, Sequence

from ..config.model import BenchmarkSpec
from ..config.setup import Setup
from ..metrics.summary import scaling_curve
from .single import DEFAULT_REPEATS, run_single

DEFAULT_COUNTS = (1, 2, 4, 8)
PROFILED_STEPS = 3


def run_scaling(
    setup: Setup,
    counts: Sequence[int] = DEFAULT_COUNTS,
    repeats: int = DEFAULT_REPEATS,
    spec: Optional[BenchmarkSpec] = None,
) -> Dict[str, Any]:
    """Measure ``setup`` on each device count in ``counts``. Returns every measurement, in increasing device counts,
    and the curve."""
    counts = sorted(set(counts))
    if not counts or counts[0] < 1:
        raise ValueError(f"device counts must be positive, got {counts}")
    setup = replace(setup, profile_steps=setup.profile_steps or PROFILED_STEPS)
    measurements = [run_single(replace(setup, devices=n), repeats=repeats, spec=spec) for n in counts]
    return {"measurements": measurements, "curve": scaling_curve([m["summary"] for m in measurements])}
