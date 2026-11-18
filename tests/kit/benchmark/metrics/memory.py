"""Memory: the peaks a run recorded, for the rank that uses the most.

- ``allocated_peak``: tensors (``max_memory_allocated``);
- ``reserved_peak``: the caching allocator (``max_memory_reserved``), which decides out-of-memory;
- ``device_peak``: the reserved peak plus what the process holds outside the allocator (context, communication
  buffers), i.e. what ``nvidia-smi`` / ``npu-smi`` shows;
- ``setup_allocated``: allocated once model and optimizer are boosted;
- ``host_rss_peak``: per process, including model construction (an offloading plugin's optimizer states live here).
"""

from typing import Any, Dict, List, Optional

GiB = 1024**3


def memory_metrics(record: Dict[str, Any]) -> Dict[str, int]:
    """Each memory statistic of a completed run, the maximum over ranks."""
    ranks = record["ranks"]
    return {key: max(r["memory"][key] for r in ranks) for key in ranks[0]["memory"]}


def max_over_repeats(memories: List[Dict[str, int]]) -> Dict[str, int]:
    return {key: max(m[key] for m in memories) for key in memories[0]}


def device_used_at_start(record: Dict[str, Any]) -> Optional[int]:
    """Bytes of the busiest device in use before the run built its model, by any process; None on the CPU. It
    includes the run's own context: well above that, another process shares the device."""
    used = [r["device_used_at_start"] for r in record["ranks"] if r["device_used_at_start"] is not None]
    return max(used) if used else None
