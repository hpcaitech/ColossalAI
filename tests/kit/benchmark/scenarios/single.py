"""Single measurement: one setup, run in several repeats, each in fresh processes, so a repeat inherits nothing
(allocator state, compiled kernels, a fragmented heap) from the one before. Every other scenario is made of these."""

from dataclasses import asdict
from typing import Any, Dict, Optional

from ..config.model import BenchmarkSpec, spec_of
from ..config.plugins import CONFIGS
from ..config.setup import Setup
from ..metrics.compute import Peak, flops_per_token, lookup
from ..metrics.summary import summarize
from ..runner.measure import measure, require_completed

DEFAULT_REPEATS = 2


def run_single(setup: Setup, repeats: int = DEFAULT_REPEATS, spec: Optional[BenchmarkSpec] = None) -> Dict[str, Any]:
    """Measure ``setup`` ``repeats`` times. Returns the setup, its FLOPs per token, the device's peak, the summary of
    its metrics and the raw runs. Raises ``runner.measure.NotCompleted`` if a run ran out of memory.

    Call from a process that has not touched the device; ``spec`` as for ``runner.measure.measure``."""
    if repeats < 1:
        raise ValueError("need at least one repeat")
    # on the meta device: no memory, and this process stays off the device
    flops = flops_per_token(
        spec or spec_of(setup.entry), setup.size, setup.seq_len, setup.micro_batch, setup.gradient_checkpointing
    )
    runs = [measure(setup, spec) for _ in range(repeats)]
    require_completed(runs)
    environment = runs[0]["environment"]
    device = environment.get("device_name", environment["accelerator"])
    dtype = CONFIGS[setup.config].dtype
    # an unknown device keeps its name and no TFLOPS: the report then says that MFU is missing and why
    peak = asdict(lookup(device, dtype) or Peak(device, dtype, nominal_tflops=None, nominal_source=""))
    return {
        "setup": asdict(setup),
        "flops": flops,
        "peak": peak,
        "summary": summarize(runs, flops, peak),
        "runs": runs,
    }
