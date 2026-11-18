"""Maximum search: the largest sequence length (at micro-batch 1) or micro-batch (at a fixed sequence length) that
trains on one device, by changing that one field of the setup.

Each probe is a full run (``runner.measure``): as many full steps (forward, backward, optimizer step) as the
measurement, in a fresh process. Near the limit the allocator's high-water mark keeps rising for a while as its cache
fragments: values that trained 3 steps ran out of memory later in the 25-step measurement of a Llama-3.2-1B on H20, so
a shorter probe reports a maximum that cannot be measured. A fresh process starts with an empty allocator, so one
probe's failure cannot make the next one fail. A probe that runs out of memory means "does not fit"; any other error
stops the search.

The search doubles from the setup's value until a probe does not fit (or halves until one does), then bisects between
the largest value that fits and the smallest that does not, on multiples of ``granularity``.
"""

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

import torch

from ..config.model import SEARCHABLE, BenchmarkSpec, spec_of
from ..config.setup import Setup
from ..metrics.memory import memory_metrics
from ..runner.measure import NotCompleted, completed, measure
from .single import DEFAULT_REPEATS, run_single

# the fields a search may change; an entry may allow fewer (BenchmarkSpec.searchable)
SEARCHED = SEARCHABLE
DEFAULT_GRANULARITY = {"seq_len": 256, "micro_batch": 1}


def probe(setup: Setup, spec: Optional[BenchmarkSpec] = None) -> Dict[str, Any]:
    """Whether ``setup`` trains all its steps on its devices, with its reserved peak or why not."""
    record = measure(setup, spec)
    if not completed(record):
        return {"fits": False, "reason": record["reason"]}
    return {"fits": True, "reserved_peak": memory_metrics(record).get("reserved_peak")}


def search_max(
    setup: Setup,
    field: str,
    fits: Callable[[Setup], Dict[str, Any]],
    granularity: Optional[int] = None,
    upper: Optional[int] = None,
) -> Dict[str, Any]:
    """The largest multiple of ``granularity`` for ``setup.<field>`` (``"seq_len"`` or ``"micro_batch"``) for which
    ``fits(setup)["fits"]``, at most ``upper``; the search starts from ``setup``'s own value. Returns the maximum (None
    if not even ``granularity`` fits) and every probe in the order run."""
    if field not in SEARCHED:
        raise ValueError(f"can search {SEARCHED}, not {field!r}")
    step = granularity or DEFAULT_GRANULARITY[field]
    if upper is not None and upper < step:
        raise ValueError(f"upper bound {upper} is below the granularity {step}")
    probes: List[Dict[str, Any]] = []

    def try_value(value: int) -> bool:
        record = fits(replace(setup, **{field: value}))
        probes.append({"value": value, **record})
        return record["fits"]

    def cap(value: int) -> int:
        return value if upper is None else min(value, upper - upper % step)

    largest_fit, smallest_miss = None, None
    value = cap(max(step, getattr(setup, field) // step * step))
    if try_value(value):
        largest_fit = value
        while upper is None or value < cap(upper):
            value = cap(value * 2)
            if not try_value(value):
                smallest_miss = value
                break
            largest_fit = value
    else:
        smallest_miss = value
        while value > step:
            value = max(step, value // 2 // step * step)
            if try_value(value):
                largest_fit = value
                break
            smallest_miss = value
    while largest_fit is not None and smallest_miss is not None and smallest_miss - largest_fit > step:
        middle = (largest_fit + smallest_miss) // 2 // step * step
        if try_value(middle):
            largest_fit = middle
        else:
            smallest_miss = middle
    return {
        "field": field,
        "granularity": step,
        "upper": upper,
        "probe_steps": setup.warmup_steps + setup.timed_steps,
        "fixed": {k: getattr(setup, k) for k in SEARCHED if k != field},
        "max": largest_fit,
        "probes": probes,
    }


def default_upper(spec: BenchmarkSpec, size: str, field: str) -> Optional[int]:
    """For a sequence length: the model's ``max_position_embeddings``, read from the model built on the meta device
    (nothing is allocated). None otherwise."""
    if field != "seq_len":
        return None
    with torch.device("meta"):
        model = spec.models[size]()
    return getattr(getattr(model, "config", None), "max_position_embeddings", None)


def run_search(
    setup: Setup,
    field: str,
    granularity: Optional[int] = None,
    upper: Optional[int] = None,
    repeats: int = DEFAULT_REPEATS,
    spec: Optional[BenchmarkSpec] = None,
) -> Dict[str, Any]:
    """Search the maximum of ``field``, then measure there (``run_single``). Returns the search and the measurement.

    A probe trains as many steps as the measurement, so the measurement normally fits too; if it runs out of memory
    after all (near the limit a run is not deterministic to the last token), the next smaller value that fitted is
    measured instead and the search says so."""
    resolved = spec or spec_of(setup.entry)
    if field not in resolved.searchable:
        raise ValueError(f"{setup.entry} can search {resolved.searchable}, not {field!r}")
    if upper is None:
        upper = default_upper(resolved, setup.size, field)
    found = search_max(setup, field, lambda s: probe(s, spec), granularity=granularity, upper=upper)
    if found["max"] is None:
        raise RuntimeError(f"not even {field}={found['granularity']} fits: {found['probes']}")
    candidates = sorted({p["value"] for p in found["probes"] if p["fits"]}, reverse=True)
    for value in candidates:
        try:
            measurement = run_single(replace(setup, **{field: value}), repeats=repeats, spec=spec)
        except NotCompleted:
            found.setdefault("full_measurement_out_of_memory", []).append(value)
            continue
        found["measured_at"] = value
        found["flags"] = (
            [f"{field}={found['max']} fitted in the search but not in the measurement; measured at {value}"]
            if value != found["max"]
            else []
        )
        return {"search": found, "measurement": measurement}
    raise RuntimeError(f"no {field} that fitted the search fits the full measurement: {candidates}")
