"""Helpers shared by both kinds of run: one forward and backward (``fwd_bwd``) and training steps (``training``)."""

from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.utils._pytree import tree_map


def cast_floating(obj: Any, dtype: torch.dtype) -> Any:
    """Cast every floating-point tensor nested in ``obj`` to ``dtype``; other leaves are returned unchanged."""
    return tree_map(lambda x: x.to(dtype) if isinstance(x, Tensor) and x.is_floating_point() else x, obj)


@contextmanager
def full_precision_matmul():
    """Disable TF32 so that float32 runs are really float32 and the float64 truth is not silently degraded."""
    matmul_precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def upstream_loss(outputs: Dict[str, Tensor], grad_seed: int = 0) -> Optional[Tensor]:
    """Scalar whose backward feeds a fixed random upstream gradient to every output that requires grad.

    The gradient is drawn in float64 from ``grad_seed`` and cast per output, so runs at different precisions (and the
    run under test, which may need a scalar loss for its own ``backward``) all see the same one. ``None`` if no output
    requires grad.
    """
    generator = torch.Generator().manual_seed(grad_seed)
    loss = None
    for out in outputs.values():
        if out.requires_grad:
            upstream = torch.randn(out.shape, generator=generator, dtype=torch.float64).to(out)
            term = (out * upstream).sum()
            loss = term if loss is None else loss + term
    return loss
