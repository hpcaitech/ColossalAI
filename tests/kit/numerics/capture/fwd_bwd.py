"""One forward and backward of a function or ``nn.Module``: the runs of operator- and module-level checks."""

import copy
from typing import Any, Callable, Dict

import torch
from torch import nn

from ..compare import flatten_tensors
from ..trace import Trace
from .utils import cast_floating, full_precision_matmul, upstream_loss


def run_fwd_bwd(fn: Callable, inputs: Dict[str, Any], dtype: torch.dtype, grad_seed: int = 0, **meta) -> Trace:
    """Run ``fn(**inputs)`` in ``dtype`` and backpropagate a fixed random upstream gradient.

    If ``fn`` is an ``nn.Module`` it is deep-copied before casting, so every run starts from the same weights.
    The upstream gradient comes from ``upstream_loss(outputs, grad_seed)``, so all runs see the same one.

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
        loss = upstream_loss(outputs, grad_seed)
        if loss is not None:
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
