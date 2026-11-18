"""Comparison: two measurements whose setups differ in one field, e.g. the plugin (``config``: a ColossalAI plugin
against the plain-PyTorch FSDP2 baseline) or the implementation (fused kernels against the plain PyTorch ones), with
everything else, the device count included, the same."""

from dataclasses import fields, replace
from typing import Any, Dict, Optional

from ..config.model import BenchmarkSpec
from ..config.setup import Setup
from ..metrics.summary import comparison
from .single import DEFAULT_REPEATS, run_single

# fields a comparison cannot change: another model is not a comparison, and profiling is not part of the measurement
FIXED = ("entry", "profile_steps")


def run_compare(
    measurement: Dict[str, Any],
    field: str,
    value: Any,
    repeats: int = DEFAULT_REPEATS,
    spec: Optional[BenchmarkSpec] = None,
) -> Dict[str, Any]:
    """Measure the setup of ``measurement`` (a ``run_single`` result) with ``field`` set to ``value``. Returns the
    field, both values, the other measurement and the ratios (above 1: ``measurement`` trains faster / uses more)."""
    if field in FIXED or field not in {f.name for f in fields(Setup)}:
        raise ValueError(f"cannot compare on {field!r}")
    setup = Setup(**measurement["setup"])
    if getattr(setup, field) == value:
        raise ValueError(f"both measurements have {field}={value!r}")
    other = run_single(replace(setup, profile_steps=0, **{field: value}), repeats=repeats, spec=spec)
    return {
        "field": field,
        "ours": getattr(setup, field),
        "theirs": value,
        "measurement": other,
        "ratios": comparison(measurement["summary"], other["summary"]),
    }
