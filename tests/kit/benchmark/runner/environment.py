"""What a run records about the machine: software versions, the device, its clock and power limits, the allocator's
settings, and how much of the device was already in use when the run started."""

import os
import platform
import subprocess
from typing import Any, Dict, Optional

import torch

import colossalai
from colossalai.accelerator import get_accelerator
from tests.kit.numerics.trace import environment_meta


def device_used() -> Optional[int]:
    """Bytes of the current device in use, by any process; None on the CPU."""
    accelerator = get_accelerator()
    if accelerator.name == "cpu":
        return None
    free, total = getattr(torch, accelerator.name).mem_get_info()
    return total - free


def environment() -> Dict[str, Any]:
    accelerator = get_accelerator()
    env = environment_meta()
    env.update(colossalai=getattr(colossalai, "__version__", "unknown"), accelerator=accelerator.name)
    if accelerator.name == "cpu":
        env["cpu"] = platform.processor() or platform.machine()
        return env
    env["device_name"] = accelerator.get_device_name(0)
    env["device_count"] = accelerator.device_count()
    # the allocator's settings (e.g. expandable_segments) change how much fits
    env["alloc_conf"] = os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF" if accelerator.name == "cuda" else "PYTORCH_NPU_ALLOC_CONF", ""
    )
    if accelerator.name == "cuda":
        env["cuda"] = torch.version.cuda
        env.update(_nvidia_smi())
    elif accelerator.name == "npu":
        import torch_npu

        env["torch_npu"] = torch_npu.__version__
    return env


def _nvidia_smi() -> Dict[str, str]:
    # the clock and power limits bound the achievable throughput; a lowered limit explains a slow run
    query = {"driver_version": "driver", "clocks.max.sm": "max_sm_clock", "power.limit": "power_limit"}
    try:
        # by UUID: the process's device 0 is any physical GPU under CUDA_VISIBLE_DEVICES, nvidia-smi counts physical ones
        uuid = f"GPU-{torch.cuda.get_device_properties(torch.cuda.current_device()).uuid}"
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(query)}", "--format=csv,noheader", f"--id={uuid}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {}
    return dict(zip(query.values(), (value.strip() for value in out.split(","))))
