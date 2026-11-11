"""The largest sequence length (at micro-batch 1) or micro-batch (at a fixed sequence length) that trains on one device.

Each probe trains as many full steps (forward, backward, optimizer step) as the measurement does (warmup and timed
steps), in a fresh process. Near the limit the allocator's high-water mark keeps rising for a while as its cache
fragments: values that trained 3 steps ran out of memory later in the 25-step measurement of a Llama-3.2-1B on H20,
so a shorter probe reports a maximum that cannot be measured. A fresh process starts with an empty allocator, so one
probe's failure cannot make the next one fail. A probe that runs out of device memory means "does
not fit"; any other error stops the search.

The search doubles from the entry's default until a probe does not fit (or halves until one does), then bisects
between the largest value that fits and the smallest that does not, on multiples of ``granularity``.
"""

import json
import os
import re
import tempfile
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.multiprocessing.spawn import ProcessExitedException, ProcessRaisedException

from colossalai.testing import spawn

from . import measure, run
from .spec import BenchmarkSpec

SEARCHED = ("seq_len", "micro_batch")
DEFAULT_GRANULARITY = {"seq_len": 256, "micro_batch": 1}
# what CUDA (torch.OutOfMemoryError) and Ascend ("NPU out of memory") raise when an allocation fails
_OUT_OF_MEMORY = re.compile(r"OutOfMemoryError|out of memory", re.IGNORECASE)


class Probe:
    """Trains one setup in fresh processes, all its steps; returns whether it fitted and its reserved peak or why not."""

    def __init__(self, nproc: int = 1, spec: Optional[BenchmarkSpec] = None):
        self.nproc, self.spec = nproc, spec

    def __call__(self, setup: measure.Setup) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "probe.json")
            try:
                spawn(run._worker, self.nproc, setup=setup, spec=self.spec, path=path)
            except ProcessRaisedException as exc:
                if not _OUT_OF_MEMORY.search(str(exc)):
                    raise
                return {"fits": False, "reason": "out of device memory"}
            except ProcessExitedException as exc:
                # killed by a signal: the host ran out of memory (an offloading plugin) and the kernel stopped it
                return {"fits": False, "reason": f"process exited ({exc}); host out of memory?"}
            with open(path) as f:
                result = json.load(f)
        return {"fits": True, "reserved_peak": result["memory_bytes"].get("reserved_peak")}


def search_max(
    setup: measure.Setup,
    field: str,
    probe: Callable[[measure.Setup], Dict[str, Any]],
    granularity: Optional[int] = None,
    upper: Optional[int] = None,
) -> Dict[str, Any]:
    """The largest multiple of ``granularity`` for ``setup.<field>`` (``"seq_len"`` or ``"micro_batch"``) for which
    ``probe`` fits, at most ``upper``; the search starts from ``setup``'s own value. Returns the maximum (None if not
    even ``granularity`` fits) and every probe in the order run."""
    if field not in SEARCHED:
        raise ValueError(f"can search {SEARCHED}, not {field!r}")
    step = granularity or DEFAULT_GRANULARITY[field]
    if upper is not None and upper < step:
        raise ValueError(f"upper bound {upper} is below the granularity {step}")
    probes: List[Dict[str, Any]] = []

    def fits(value: int) -> bool:
        record = probe(replace(setup, **{field: value}))
        probes.append({"value": value, **record})
        return record["fits"]

    def cap(value: int) -> int:
        return value if upper is None else min(value, upper - upper % step)

    largest_fit, smallest_miss = None, None
    value = cap(max(step, getattr(setup, field) // step * step))
    if fits(value):
        largest_fit = value
        while upper is None or value < cap(upper):
            value = cap(value * 2)
            if not fits(value):
                smallest_miss = value
                break
            largest_fit = value
    else:
        smallest_miss = value
        while value > step:
            value = max(step, value // 2 // step * step)
            if fits(value):
                largest_fit = value
                break
            smallest_miss = value
    while largest_fit is not None and smallest_miss is not None and smallest_miss - largest_fit > step:
        middle = (largest_fit + smallest_miss) // 2 // step * step
        if fits(middle):
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
    setup: measure.Setup,
    field: str,
    granularity: Optional[int] = None,
    upper: Optional[int] = None,
    repeats: int = run.DEFAULT_REPEATS,
    spec: Optional[BenchmarkSpec] = None,
    command: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Search the maximum of ``field`` on one device, then measure throughput and memory there with the full protocol;
    the report of that measurement carries the search under ``"search"``.

    The maximum is the largest value whose probe fitted. A probe trains as many steps as the measurement, so the
    measurement normally fits too; if it runs out of memory after all (near the limit a run is not deterministic to
    the last token), the next smaller value that fitted is measured instead and the report says so."""
    found = search_max(setup, field, Probe(spec=spec), granularity=granularity, upper=upper)
    if found["max"] is None:
        raise RuntimeError(f"not even {field}={found['granularity']} fits on one device: {found['probes']}")
    candidates = sorted({p["value"] for p in found["probes"] if p["fits"]}, reverse=True)
    for value in candidates:
        try:
            result = run.run_benchmark(replace(setup, **{field: value}), repeats=repeats, spec=spec, command=command)
        except ProcessRaisedException as exc:
            if not _OUT_OF_MEMORY.search(str(exc)):
                raise
            found.setdefault("full_measurement_out_of_memory", []).append(value)
            continue
        found["measured_at"] = value
        result["search"] = found
        if value != found["max"]:
            result["flags"].append(
                f"{field}={found['max']} fitted in the search but not in the measurement; measured at {value}"
            )
        return result
    raise RuntimeError(f"no {field} that fitted the search fits the full measurement: {candidates}")
