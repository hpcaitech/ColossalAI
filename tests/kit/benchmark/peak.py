"""Peak dense matmul throughput of each device, the denominator of MFU.

Two numbers per device and dtype, reported side by side:

- ``nominal``: the vendor's dense (no sparsity) figure, the industry's denominator, so MFU compares with published
  numbers. Each entry names its source.
- ``measured``: the best median of square matmuls (n = 4096, 8192, 16384, device-event timed, TF32 / HF32 off for
  fp32) on our machines. A device that cannot sustain its nominal figure (the Ascend 910B2 is power-capped at 400 W
  and clocks down under load) is far from 100% MFU even when its matmuls are as fast as they get; MFU against the
  measured peak separates that from the framework's own overhead.

Devices are recognized by ``accelerator.get_device_name()``. A device or dtype missing here gives no MFU, which the
report says, rather than a guessed one.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Peak:
    device: str
    dtype: str
    nominal_tflops: Optional[float]
    nominal_source: str
    measured_tflops: Optional[float] = None
    measured_source: str = ""


_H20_NOMINAL = (
    "NVIDIA has no public H20 datasheet; figure consistent across third-party listings and press (dense Tensor Core)"
)
_H20_MEASURED = "one H20 (driver 595.71.05, SM clock steady at 1980 MHz, ~488 W of 500 W), torch 2.11, 2026-11-11"
_910B2_NOMINAL = (
    "Huawei, Atlas 800T A2 training server user guide (version 20, 2025-11-17), table 3-1, peak dense compute "
    "provided stably: {figures} depending on the processor; the higher is the 910B2 (24 cube cores against the "
    "910B3's 20 at the same 1800 MHz, CANN platform config)"
)
_910B2_MEASURED = (
    "two 910B2 servers (CANN 9.1.0, torch_npu 2.13.0rc1), 2026-11-11; power at the 400 W cap, AI core clock down "
    "from 1800 to 1400-1600 MHz"
)
_910B2_FP16 = _910B2_NOMINAL.format(figures="313 or 376 TFLOPS@FP16")

PEAKS: Tuple[Peak, ...] = (
    Peak("H20", "bf16", 148.0, _H20_NOMINAL, 140.6, _H20_MEASURED),
    Peak("H20", "fp16", 148.0, _H20_NOMINAL, 140.5, _H20_MEASURED),
    Peak("H20", "fp32", 44.0, _H20_NOMINAL + "; FP32 without Tensor Cores", 32.5, _H20_MEASURED),
    Peak("Ascend910B2", "fp16", 376.0, _910B2_FP16, 260.5, _910B2_MEASURED),
    Peak(
        "Ascend910B2", "bf16", 376.0, _910B2_FP16 + "; BF16 is not listed, taken equal to FP16", 290.8, _910B2_MEASURED
    ),
    Peak("Ascend910B2", "fp32", 99.0, _910B2_NOMINAL.format(figures="82 or 99 TFLOPS@FP32"), 75.3, _910B2_MEASURED),
)


def device_key(device_name: str) -> Optional[str]:
    """The ``PEAKS`` device that a ``get_device_name()`` string names, or None.

    ``NVIDIA H20`` and its larger-memory variant have the same compute; Ascend names are matched exactly, as the
    910B variants differ in compute."""
    if "H20" in device_name.replace("-", " ").split():
        return "H20"
    if device_name in {p.device for p in PEAKS}:
        return device_name
    return None


def lookup(device_name: str, dtype: str) -> Optional[Peak]:
    key = device_key(device_name)
    return next((p for p in PEAKS if p.device == key and p.dtype == dtype), None)
