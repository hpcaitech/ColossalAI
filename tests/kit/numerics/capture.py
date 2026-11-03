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


# Switches, named as dotted paths under ``torch``, that let a backend compute a float32 operation in fewer bits.
# They differ per backend and are not all off by default: Ascend enables HF32 for convolution, which on its own makes
# a float32 Conv3d about 800x less accurate than the same convolution on CPU.
REDUCED_PRECISION_FLAGS = (
    "backends.cudnn.allow_tf32",
    "npu.matmul.allow_hf32",
    "npu.conv.allow_hf32",
)


def _resolve_flag(path):
    """Return the ``(holder, attribute)`` that ``path`` names under ``torch``, or ``None`` if this build lacks it."""
    holder = torch
    *parents, attribute = path.split(".")
    for part in parents:
        holder = getattr(holder, part, None)
        if holder is None:
            return None
    return (holder, attribute) if hasattr(holder, attribute) else None


@contextmanager
def full_precision_matmul():
    """Keep float32 runs in float32, so the float64 truth and the baseline error are not silently degraded.

    Every switch in ``REDUCED_PRECISION_FLAGS`` that this build has is cleared and restored afterwards, because a
    backend left to its defaults may compute a nominally float32 matmul or convolution in fewer bits. Disabling only
    the switches of one backend would make this helper silently useless on the others.
    """
    matmul_precision = torch.get_float32_matmul_precision()
    saved = []
    for path in REDUCED_PRECISION_FLAGS:
        found = _resolve_flag(path)
        if found is not None:
            holder, attribute = found
            saved.append((holder, attribute, getattr(holder, attribute)))
    torch.set_float32_matmul_precision("highest")
    for holder, attribute, _ in saved:
        setattr(holder, attribute, False)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
        for holder, attribute, value in saved:
            setattr(holder, attribute, value)


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
