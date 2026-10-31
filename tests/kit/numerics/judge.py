"""Pass/fail rules for numerical alignment.

The baseline-error rule follows PyTorch's TorchInductor accuracy check (``same()`` in ``torch/_dynamo/utils.py``):
the baseline error is how far the reference implementation itself deviates from a float64 ground truth at the
precision being tested, and the implementation under test may deviate at most ``multiplier`` times as much.
"""

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from .compare import TensorMetrics, as_float64, flatten_tensors, tensor_metrics

LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def error_multiplier(dtype: torch.dtype, numel: int, small_tensor_multiplier: bool = False) -> float:
    """Allowed ratio between the test error and the baseline error, as chosen by PyTorch.

    3 for fp16/bf16 (2 gave false alarms under AMP), 2 otherwise; small tensors are noisier and get at least 3,
    or 8-10 with ``small_tensor_multiplier``.
    """
    multiplier = 3.0 if dtype in LOW_PRECISION_DTYPES else 2.0
    if small_tensor_multiplier and numel <= 10:
        return 10.0
    if small_tensor_multiplier and numel < 1000:
        return 8.0
    if numel < 1000:
        return max(multiplier, 3.0)
    return multiplier


@dataclass
class Entry:
    name: str
    passed: bool
    reason: str
    baseline_error: float = math.nan
    test_error: float = math.nan
    allowed_error: float = math.nan
    metrics: Optional[TensorMetrics] = None

    @property
    def ratio(self) -> float:
        if self.baseline_error == 0:
            return 0.0 if self.test_error == 0 else math.inf
        return self.test_error / self.baseline_error


@dataclass
class AlignmentReport:
    entries: List[Entry] = field(default_factory=list)

    @property
    def failures(self) -> List[Entry]:
        return [e for e in self.entries if not e.passed]

    @property
    def passed(self) -> bool:
        return not self.failures

    def format(self, limit: int = 20) -> str:
        rows = sorted(self.entries, key=lambda e: (e.passed, -e.ratio if not math.isnan(e.ratio) else 0.0))
        lines = [
            f"{len(self.entries) - len(self.failures)}/{len(self.entries)} tensors aligned",
            f"{'tensor':<56} {'baseline':>9} {'test_err':>9} {'ratio':>8}  verdict",
        ]
        for e in rows[:limit]:
            verdict = "PASS" if e.passed else f"FAIL ({e.reason})"
            lines.append(f"{e.name:<56} {e.baseline_error:9.2e} {e.test_error:9.2e} {e.ratio:8.2f}  {verdict}")
        if len(rows) > limit:
            lines.append(f"... {len(rows) - limit} more")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()

    def assert_pass(self) -> None:
        assert self.passed, "numerical alignment failed\n" + self.format()


def judge(
    test: Any,
    ref: Any,
    truth: Any,
    tol: float = 1e-4,
    small_tensor_multiplier: bool = False,
) -> AlignmentReport:
    """Judge ``test`` against a float64 ``truth``, allowing ``multiplier`` times the baseline error.

    The baseline error is the error of ``ref`` against ``truth``: what correct code naturally gets wrong at this
    precision.

    Args:
        test: Outputs of the implementation under test, at the precision being tested.
        ref: Outputs of the reference implementation run with the same precision recipe as ``test``.
        truth: Outputs of the reference implementation in float64.
        tol: Fast-path tolerance: ``test`` within ``allclose(ref, atol=tol, rtol=tol)`` passes outright;
            ``tol / 10`` is also added to the allowed error, as in PyTorch.

    Each argument may be a tensor or any nesting of dicts / lists / tuples / ``ModelOutput``; tensors are matched
    by their flattened names (see ``flatten_tensors``).
    """
    test_flat, ref_flat, truth_flat = flatten_tensors(test), flatten_tensors(ref), flatten_tensors(truth)
    report = AlignmentReport()
    for name in sorted(set(truth_flat) - set(test_flat)):
        report.entries.append(Entry(name, False, "missing in test"))
    for name in sorted(set(test_flat) - set(truth_flat)):
        report.entries.append(Entry(name, False, "missing in truth"))
    for name in sorted(set(truth_flat) & set(test_flat)):
        report.entries.append(
            _judge_one(name, test_flat[name], ref_flat.get(name), truth_flat[name], tol, small_tensor_multiplier)
        )
    return report


def _judge_one(name, test, ref, truth, tol, small_tensor_multiplier) -> Entry:
    if test.shape != truth.shape:
        return Entry(name, False, f"shape {tuple(test.shape)} vs {tuple(truth.shape)}")
    if not test.is_floating_point():
        equal = bool(torch.equal(test.cpu(), truth.cpu()))
        return Entry(name, equal, "exact" if equal else "non-float tensors differ", 0.0, 0.0, 0.0)
    if ref is None:
        return Entry(name, False, "missing in ref")
    if ref.shape != truth.shape:
        return Entry(name, False, f"ref shape {tuple(ref.shape)} vs {tuple(truth.shape)}")

    test64, ref64, truth64 = as_float64(test), as_float64(ref), as_float64(truth)
    metrics = tensor_metrics(test64, truth64)
    baseline_error, test_error = tensor_metrics(ref64, truth64).rmse, metrics.rmse
    multiplier = error_multiplier(test.dtype, test.numel(), small_tensor_multiplier)
    allowed = multiplier * baseline_error + tol / 10.0
    entry = Entry(name, False, "", baseline_error, test_error, allowed, metrics)

    if not torch.isfinite(truth64).all():
        entry.reason = "non-finite truth, rerun the reference in higher precision"
    elif not torch.isfinite(test64).all():
        entry.reason = "non-finite test output"
    elif torch.allclose(test64, ref64, atol=tol, rtol=tol):
        entry.passed, entry.reason = True, "allclose to ref"
    elif math.isnan(baseline_error):
        entry.reason = "non-finite ref output"
    elif test_error <= allowed:
        entry.passed, entry.reason = True, "within baseline"
    else:
        entry.reason = f"> {multiplier:g}x baseline error"
    return entry


def assert_aligned(test: Any, ref: Any, truth: Any, **kwargs) -> AlignmentReport:
    """``judge`` and raise ``AssertionError`` with a per-tensor table if anything is not aligned."""
    report = judge(test, ref, truth, **kwargs)
    report.assert_pass()
    return report
