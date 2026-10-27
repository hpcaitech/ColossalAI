"""Helpers to produce the three runs (float64 truth, reference, test) for operator- and module-level checks."""

import copy
from contextlib import contextmanager
from typing import Any, Callable, Dict

import torch
from torch import Tensor, nn
from torch.utils._pytree import tree_map

from .compare import flatten_tensors
from .trace import Trace


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


def run_fwd_bwd(fn: Callable, inputs: Dict[str, Any], dtype: torch.dtype, grad_seed: int = 0, **meta) -> Trace:
    """Run ``fn(**inputs)`` in ``dtype`` and backpropagate a fixed random upstream gradient.

    If ``fn`` is an ``nn.Module`` it is deep-copied before casting, so every run starts from the same weights.
    The upstream gradient is drawn in float64 from ``grad_seed`` and cast per output, so all runs see the same one.

    Returns a ``Trace`` with ``{"out/...": output, "grad/<input name>": input grad, "grad/param/<param name>": param
    grad}``; ``meta`` (e.g. ``role="truth"``) is recorded alongside dtype, function name and seed.
    """
    fn_name = type(fn).__name__ if isinstance(fn, nn.Module) else getattr(fn, "__qualname__", repr(fn))
    if isinstance(fn, nn.Module):
        fn = copy.deepcopy(fn).to(dtype)
    leaves = {}
    for name, value in cast_floating(inputs, dtype).items():
        if isinstance(value, Tensor) and value.is_floating_point():
            value = value.detach().clone().requires_grad_(True)
        leaves[name] = value

    with full_precision_matmul():
        outputs = flatten_tensors(fn(**leaves), "out")
        generator = torch.Generator().manual_seed(grad_seed)
        loss = 0
        for out in outputs.values():
            if out.requires_grad:
                upstream = torch.randn(out.shape, generator=generator, dtype=torch.float64).to(out)
                loss = loss + (out * upstream).sum()
        if isinstance(loss, Tensor):
            loss.backward()

    results = {name: out.detach() for name, out in outputs.items()}
    for name, value in leaves.items():
        if isinstance(value, Tensor) and value.grad is not None:
            results[f"grad/{name}"] = value.grad
    if isinstance(fn, nn.Module):
        for name, param in fn.named_parameters():
            if param.grad is not None:
                results[f"grad/param/{name}"] = param.grad
    return Trace.capture(results, fn=fn_name, dtype=str(dtype), grad_seed=grad_seed, **meta)
