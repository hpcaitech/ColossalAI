"""The device timeline of profiled steps: when each step ran, and when communication and other kernels ran in it.

Kept as time intervals (overlapping kernels merged into one interval), not as shares: ``metrics.parallel`` derives the
communication shares from them. CUDA only: the Ascend profiler is a different API (``torch_npu.profiler``) and is not
read yet.
"""

import json
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

# profiler range of step i (warmup included): f"{STEP_ANNOTATION}{i}"
STEP_ANNOTATION = "benchmark/step"
# what NCCL and HCCL name their kernels
COMMUNICATION_KERNELS = ("nccl", "hccl")

Span = Tuple[float, float]


def union(spans: Sequence[Span]) -> List[Span]:
    """``spans`` sorted, with overlapping and touching ones merged."""
    merged: List[List[float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(span) for span in merged]


def spans_from_trace(events: List[Dict[str, Any]]) -> Dict[str, List[Span]]:
    """From a Chrome trace: each step's range on the device timeline (its ``STEP_ANNOTATION``), and the device time in
    which communication kernels and other kernels ran, in microseconds."""
    steps = [
        (e["ts"], e["ts"] + e["dur"])
        for e in events
        if e.get("cat") == "gpu_user_annotation" and e.get("name", "").startswith(STEP_ANNOTATION)
    ]
    if not steps:
        raise ValueError("the trace has no profiled steps")
    communication, compute = [], []
    for e in events:
        if e.get("cat") != "kernel":
            continue
        span = (e["ts"], e["ts"] + e["dur"])
        is_communication = any(word in e.get("name", "").lower() for word in COMMUNICATION_KERNELS)
        (communication if is_communication else compute).append(span)
    return {"steps": sorted(steps), "communication": union(communication), "compute": union(compute)}


def profile(device: str, run_steps: Callable[[], None]) -> Optional[Dict[str, List[Span]]]:
    """Run ``run_steps()`` under the profiler and return this rank's ``spans_from_trace``; on a device whose profiler
    is not read, run them unprofiled and return None."""
    if device != "cuda":
        run_steps()
        return None
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "trace.json")
        with torch.profiler.profile(activities=activities) as profiler:
            run_steps()
        profiler.export_chrome_trace(path)
        with open(path) as f:
            events = json.load(f)["traceEvents"]
    return spans_from_trace(events)
