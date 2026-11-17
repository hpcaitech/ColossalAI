"""Checks: when a measurement's numbers must not be used, as warnings the report shows above its results."""

from typing import Any, Dict, List

from .compute import FLOPS_MISMATCH_LIMIT
from .memory import GiB

# two repeats of the same setup whose median step times differ by more than this are flagged: the machine was not
# stable enough for the numbers to be compared with anything
REPEAT_SPREAD_LIMIT = 0.05
# device memory already in use when a run starts, as a share of the device, above which another process shares it.
# The run's own context is a few hundred MiB on CUDA; the Ascend driver's baseline is about 5% of a 910B2
BUSY_DEVICE_LIMIT = 0.10


def flags(summary: Dict[str, Any], flops: Dict[str, Any]) -> List[str]:
    """Warnings for a ``summary.summarize`` result measured with ``flops``."""
    found = []
    spread = summary["repeat_spread"]
    if spread > REPEAT_SPREAD_LIMIT:
        found.append(
            f"repeats differ by {spread:.1%} in median step time (limit {REPEAT_SPREAD_LIMIT:.0%}): the machine was "
            "not stable, do not compare these numbers"
        )
    if not all(r["loss_finite"] for r in summary["repeats"]):
        found.append("a loss was not finite: the steps did not train, the timing may not be representative")
    total = summary["memory_bytes"].get("device_total")
    used = [r["device_used_at_start"] for r in summary["repeats"] if r["device_used_at_start"] is not None]
    if used and total and max(used) > BUSY_DEVICE_LIMIT * total:
        found.append(
            f"{max(used) / GiB:.1f} GiB of a device were in use when the run started: another process shared it, the "
            "numbers are not reliable"
        )
    mfu = summary["utilization"]["mfu_nominal"]
    if mfu is not None and mfu > 1:
        found.append(
            f"MFU is {mfu:.0%}, above the device's peak: the FLOPs per token overstate the work done (e.g. attention "
            "that the kernel skips), MFU is not reliable"
        )
    if flops["difference"] is not None and abs(flops["difference"]) > FLOPS_MISMATCH_LIMIT:
        found.append(
            f"the FLOPs formula and the counted FLOPs differ by {flops['difference']:+.1%} (limit "
            f"{FLOPS_MISMATCH_LIMIT:.0%}): one of them does not describe the model, MFU is not reliable"
        )
    return found
