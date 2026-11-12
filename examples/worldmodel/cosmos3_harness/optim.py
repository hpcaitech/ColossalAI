"""AdamW arms shared by the training entry point and the operator benchmark.

Kept free of the modeling imports so ``bench_ops.py`` can measure the arms without loading the policy stack.
"""

from __future__ import annotations

from typing import Sequence

import torch


def build_adamw(
    params,
    learning_rate: float,
    backend: str = "auto",
    mode: str = "auto",
    *,
    device_type: str | None = None,
    betas: Sequence[float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
):
    """Build one of the benchmark arms; all of them share the same update math.

    Upstream ``torch.optim.AdamW`` only turns ``foreach`` on by default for the devices on its supported list
    (CUDA/XPU), so unless ``torch_npu`` patches that list, NPU takes the per-tensor path: one launch per
    parameter, hundreds per ZeRO-2 step.

    - ``plain``: PyTorch's default, per-tensor kernels.
    - ``foreach``: ``torch._foreach_*`` multi-tensor kernels (implemented by ``torch_npu``).
    - ``fused``: ``torch_npu.optim.NpuFusedAdamW``. Experimental: it merges parameter state once, so it must be
      checked against ``foreach`` on the target box before being trusted under a wrapper that rewrites
      ``param_groups[...]["params"]`` each step, as ``LowLevelZeroOptimizer`` does.
    - ``auto``: ``foreach`` on the target NPU device, ``plain`` elsewhere.

    ``backend`` remains for call compatibility; attention never selects the optimizer device.
    ``device_type`` names the target device when constructing CPU parameters before boosting.
    """
    # backend is retained as a compatibility keyword; attention does not select the device.
    params = list(params)
    parameter_devices = {
        p.device.type for group in params for p in (group["params"] if isinstance(group, dict) else [group])
    }
    if device_type is None:
        if len(parameter_devices) != 1:
            raise ValueError("AdamW needs parameters on one device type or an explicit target device_type")
        device_type = next(iter(parameter_devices))
    if mode == "auto":
        mode = "foreach" if device_type == "npu" else "plain"
    kwargs = dict(lr=learning_rate, betas=tuple(betas), eps=eps, weight_decay=weight_decay)
    if mode == "fused":
        if device_type != "npu":
            raise ValueError("The fused AdamW arm requires an NPU device")
        try:
            from torch_npu.optim import NpuFusedAdamW
        except ImportError as exc:
            raise ValueError("fused AdamW requires torch_npu.optim.NpuFusedAdamW") from exc
        return NpuFusedAdamW(params, **kwargs)
    if mode == "foreach":
        return torch.optim.AdamW(params, foreach=True, **kwargs)
    if mode == "plain":
        return torch.optim.AdamW(params, foreach=False, **kwargs)
    raise ValueError(f"unknown AdamW mode {mode!r}; expected auto, plain, foreach or fused")
