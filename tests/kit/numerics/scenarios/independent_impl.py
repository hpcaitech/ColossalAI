"""Scenario ① and ⑤: the code under test against an independent reference implementation.

The reference shares no code with what is tested (a plain implementation for a fused kernel, HuggingFace or an
official repository for a model), so a bug cannot hide in both. Three runs from the same inputs and weights: the
reference in float64 (``truth``), the reference at the tested precision (``ref``) and the code under test
(``test``), judged with the baseline-error rule (``judge``). Two granularities:

- ``check_fwd_bwd``: one forward and backward of a function or module (``capture.fwd_bwd``).
- ``check_training``: a model_zoo entry trained through a booster plugin (``capture.training``). ``steps=1``
  compares outputs, gradients and the weights after one optimizer step; ``steps=N`` adds the loss of every step.

For training, the precision recipes must match, guarded three ways:

1. ``recipe_for`` derives the reference recipe from the plugin configuration and refuses anything unverified.
2. The dtypes both runs actually used are compared (``RecipeMismatch`` before any numbers are judged).
3. ``blank_control`` runs a configuration in which the plugin changes no arithmetic; every ratio must be 1.

The reference is also run twice (as PyTorch does before its accuracy checks): tensors that differ between the two
runs come from nondeterministic kernels, and the worse of the two runs is used as ``ref`` for them.
"""

from typing import Any, Callable, Dict, List, Tuple, Union

import torch

from colossalai.accelerator import get_accelerator
from colossalai.booster.mixed_precision import MixedPrecision
from colossalai.booster.plugin import LowLevelZeroPlugin, Plugin, TorchDDPPlugin

from ..capture.fwd_bwd import run_fwd_bwd
from ..capture.recipes import FLOAT64, recipe_for
from ..capture.training import default_optimizer, prepare, run_boosted, run_reference
from ..compare import as_float64
from ..judge import AlignmentReport, judge
from ..trace import Trace


class RecipeMismatch(AssertionError):
    pass


def training_traces(
    entry: Union[str, tuple],
    plugin: Plugin,
    mixed_precision: Union[str, MixedPrecision, None] = None,
    steps: int = 1,
    seed: int = 0,
    grad_seed: int = 0,
    optimizer_fn: Callable = default_optimizer,
) -> Tuple[Trace, Trace, Trace]:
    """Run ``entry`` (a model_zoo name or entry tuple) and return ``(test, ref, truth)`` for ``judge``.

    Must be called with ``torch.distributed`` initialized (e.g. inside ``colossalai.testing.spawn``).
    Raises ``UnsupportedRecipe`` for unverified configurations and ``RecipeMismatch`` if the dtypes differ.
    """
    recipe = recipe_for(plugin, mixed_precision)
    s = prepare(entry, steps, seed)
    truth = run_reference(s, FLOAT64, optimizer_fn, grad_seed, role="truth")
    ref = run_reference(s, recipe, optimizer_fn, grad_seed, role="ref")
    ref_again = run_reference(s, recipe, optimizer_fn, grad_seed, role="ref")
    ref = _worse_of(ref, ref_again, truth)
    test = run_boosted(s, plugin, mixed_precision, optimizer_fn, grad_seed, role="test")
    mismatches = dtype_mismatches(test, ref)
    if mismatches:
        raise RecipeMismatch(
            f"test and ref used different dtypes, so they do not follow the same precision recipe "
            f"(ref: {ref.meta['recipe']}):\n" + "\n".join(mismatches)
        )
    return test, ref, truth


def check_training(*args, **kwargs) -> AlignmentReport:
    """``judge(*training_traces(...))``."""
    return judge(*training_traces(*args, **kwargs))


def check_fwd_bwd(
    test_fn: Callable,
    ref_fn: Callable,
    inputs: Dict[str, Any],
    dtype: torch.dtype,
    grad_seed: int = 0,
    **judge_kwargs,
) -> AlignmentReport:
    """Judge one forward and backward of ``test_fn`` against ``ref_fn`` at ``dtype``, with ``ref_fn`` in float64 as
    the truth. Both take ``inputs`` as keyword arguments; modules are copied, so give ``test_fn`` the reference weights
    first (``load_state_dict``). ``judge_kwargs`` go to ``judge`` (``tol``, ``small_tensor_multiplier``)."""
    truth = run_fwd_bwd(ref_fn, inputs, torch.float64, grad_seed, role="truth")
    ref = run_fwd_bwd(ref_fn, inputs, dtype, grad_seed, role="ref")
    test = run_fwd_bwd(test_fn, inputs, dtype, grad_seed, role="test")
    return judge(test, ref, truth, **judge_kwargs)


def dtype_mismatches(test: Trace, ref: Trace) -> List[str]:
    """Dtype records present in both runs that differ, as ``"<key>: test <dtypes> / ref <dtypes>"``."""
    test_dtypes, ref_dtypes = test.meta["dtypes"], ref.meta["dtypes"]
    return [
        f"{key}: test {test_dtypes[key]} / ref {ref_dtypes[key]}"
        for key in sorted(set(test_dtypes) & set(ref_dtypes))
        if test_dtypes[key] != ref_dtypes[key]
    ]


def blank_control(entry: Union[str, tuple], steps: int = 1, seed: int = 0, slack: float = 1.05) -> AlignmentReport:
    """Defense 3: check the reference implementation itself before trusting any comparison.

    Uses a configuration in which the plugin changes no arithmetic: one device, ``LowLevelZeroPlugin(stage=1,
    precision="bf16")`` (``TorchDDPPlugin()`` in float32 on CPU, where LowLevelZero cannot run), the same AdamW on
    both sides. Every tensor's error must then be within ``slack`` times the baseline error.
    """
    if get_accelerator().name == "cpu":
        plugin = TorchDDPPlugin()
    else:
        plugin = LowLevelZeroPlugin(stage=1, precision="bf16")
    report = check_training(entry, plugin, steps=steps, seed=seed)
    # a failed entry (missing tensor, shape) counts too; NaN errors compare False, so check passed explicitly
    off = [e for e in report.entries if not e.passed or e.test_error > slack * e.baseline_error]
    if off:
        names = ", ".join(e.name for e in off)
        raise AssertionError(
            f"blank control: the reference does not reproduce {type(plugin).__name__} where it changes no "
            f"arithmetic ({names})\n{report.format()}"
        )
    return report


def _worse_of(ref: Trace, ref_again: Trace, truth: Trace) -> Trace:
    tensors, nondeterministic = {}, []
    for name, a in ref.items():
        b = ref_again[name]
        if torch.equal(a, b):
            tensors[name] = a
            continue
        nondeterministic.append(name)
        if name in truth:
            t = as_float64(truth[name])
            a_err = (as_float64(a) - t).pow(2).mean()
            b_err = (as_float64(b) - t).pow(2).mean()
            tensors[name] = a if a_err >= b_err else b
        else:
            tensors[name] = a
    return Trace(tensors, dict(ref.meta, nondeterministic=nondeterministic))
