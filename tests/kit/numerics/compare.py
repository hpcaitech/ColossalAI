"""Tensor-level comparison for numerical alignment tests.

All metrics are computed in float64 on CPU so that the measurement itself adds no rounding error.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import Tensor


@dataclass
class TensorMetrics:
    rmse: float
    max_abs: float
    rel_l2: float
    cosine: float


def flatten_tensors(obj: Any, prefix: str = "") -> Dict[str, Tensor]:
    """Flatten tensors nested in mappings (dicts, HF ``ModelOutput``, ``Trace``), lists and tuples into ``{"a/b/0": tensor}``.

    Non-tensor leaves such as ``None`` are dropped.
    """
    if isinstance(obj, Tensor):
        return {prefix or "output": obj}
    if isinstance(obj, Mapping):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    else:
        return {}
    flat = {}
    for key, value in items:
        flat.update(flatten_tensors(value, f"{prefix}/{key}" if prefix else str(key)))
    return flat


def as_float64(tensor: Tensor) -> Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float64)


def tensor_metrics(actual: Tensor, expected: Tensor) -> TensorMetrics:
    """Distance of ``actual`` from ``expected``. NaN or inf in either input propagates into the metrics."""
    a, e = as_float64(actual).flatten(), as_float64(expected).flatten()
    if a.numel() == 0:
        return TensorMetrics(0.0, 0.0, 0.0, 1.0)
    diff = a - e
    a_norm, e_norm, diff_norm = a.norm().item(), e.norm().item(), diff.norm().item()
    if e_norm > 0:
        rel_l2 = diff_norm / e_norm
    else:
        rel_l2 = 0.0 if diff_norm == 0 else math.inf
    if a_norm > 0 and e_norm > 0:
        cosine = (a @ e).item() / (a_norm * e_norm)
    else:
        cosine = 1.0 if a_norm == e_norm else 0.0
    return TensorMetrics(
        rmse=diff.pow(2).mean().sqrt().item(),
        max_abs=diff.abs().max().item(),
        rel_l2=rel_l2,
        cosine=cosine,
    )
