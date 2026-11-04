"""Scenario ① and ⑤: the code under test against an independent reference implementation.

The reference shares no code with what is tested (a plain implementation for a fused kernel, HuggingFace or an
official repository for a model), so a bug cannot hide in both. Three runs from the same inputs and weights: the
reference in float64 (``truth``), the reference at the tested precision (``ref``) and the code under test
(``test``), judged with the baseline-error rule (``judge``). Two granularities:

- ``check_fwd_bwd``: one forward and backward of a function or module (``capture.fwd_bwd``).
- ``check_training``: a model_zoo entry trained through a booster plugin (``capture.training``). ``steps=1``
  compares outputs, gradients and the weights after one optimizer step; ``steps=N`` adds the loss of every step.
- ``check_optimizer``: an optimizer's update rule against a reference optimizer, from the same weights and gradients
  (``capture.optimizer_step``).

For training, the precision recipes must match, guarded three ways:

1. ``recipe_for`` derives the reference recipe from the plugin configuration and refuses anything unverified.
2. The dtypes and device types both runs actually used are compared (``RecipeMismatch`` before any numbers are
   judged).
3. ``blank_control`` runs a configuration in which the plugin changes no arithmetic; every ratio must be 1.

The reference is also run twice (as PyTorch does before its accuracy checks): tensors that differ between the two
runs come from nondeterministic kernels, and the worse of the two runs is used as ``ref`` for them.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from colossalai.accelerator import get_accelerator
from colossalai.booster.mixed_precision import MixedPrecision
from colossalai.booster.plugin import LowLevelZeroPlugin, Plugin, TorchDDPPlugin

from ..capture.cache import cached_run, reference_key
from ..capture.fwd_bwd import run_fwd_bwd
from ..capture.optimizer_step import run_optimizer
from ..capture.plugins import FirstMomentIsGradient, exposes_gradients
from ..capture.recipes import FLOAT64, recipe_for
from ..capture.training import default_optimizer, prepare, run_boosted, run_reference
from ..capture.utils import count_calls
from ..compare import as_float64
from ..judge import AlignmentReport, Entry, diagnose_routing, judge, require_gradients
from ..spec import Routers
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
    cache_dir: Optional[str] = None,
) -> Tuple[Trace, Trace, Trace]:
    """Run ``entry`` (a model_zoo name or entry tuple) and return ``(test, ref, truth)`` for ``judge``.

    Must be called with ``torch.distributed`` initialized (e.g. inside ``colossalai.testing.spawn``).
    Raises ``UnsupportedRecipe`` for unverified configurations and ``RecipeMismatch`` if the dtypes or device types
    differ.
    For Gemini and FSDP, whose gradients are read from Adam's first moment, both sides run ``optimizer_fn`` with
    ``beta1 = 0`` (``capture.plugins``).

    With ``cache_dir``, the reference runs (``truth`` and ``ref``) are saved there and reused while nothing that
    determines them changes (``capture.cache``); the float64 truth is shared by every configuration of the entry. Use
    it for real weights, where the float64 run is the slow part. The model under test always runs.
    """
    recipe = recipe_for(plugin, mixed_precision)
    if not exposes_gradients(plugin):
        optimizer_fn = FirstMomentIsGradient(optimizer_fn)
    s = prepare(entry, steps, seed)
    truth = cached_run(
        cache_dir,
        s.name,
        "truth",
        reference_key(s, optimizer_fn, grad_seed),
        lambda: run_reference(s, FLOAT64, optimizer_fn, grad_seed, role="truth"),
    )

    def run_ref() -> Trace:
        ref = run_reference(s, recipe, optimizer_fn, grad_seed, role="ref")
        ref_again = run_reference(s, recipe, optimizer_fn, grad_seed, role="ref")
        return _worse_of(ref, ref_again, truth)

    ref = cached_run(cache_dir, s.name, "ref", reference_key(s, optimizer_fn, grad_seed, recipe), run_ref)
    test = run_boosted(s, plugin, mixed_precision, optimizer_fn, grad_seed, role="test")
    mismatches = dtype_mismatches(test, ref)
    if mismatches:
        raise RecipeMismatch(
            f"test and ref used different dtypes or devices, so they do not follow the same precision recipe "
            f"(ref: {ref.meta['recipe']}):\n" + "\n".join(mismatches)
        )
    return test, ref, truth


def check_training(*args, **kwargs) -> AlignmentReport:
    """``judge(*training_traces(...))``, failing what breaks the entry's parameter contract in any run
    (``require_gradients``): a trainable parameter left without a gradient unless ``NumericsSpec.unused_params`` names
    it, one the optimizer does not hold, a tie that differs from ``NumericsSpec.tied_params``. With
    ``NumericsSpec.routers``, routing flips are counted (``diagnose_routing``)."""
    test, ref, truth = training_traces(*args, **kwargs)
    report = require_gradients(judge(test, ref, truth), test=test, ref=ref, truth=truth)
    return diagnose_routing(report, test, ref, truth)


def check_fwd_bwd(
    test_fn: Callable,
    ref_fn: Callable,
    inputs: Dict[str, Any],
    dtype: torch.dtype,
    grad_seed: int = 0,
    unused_params: Iterable[str] = (),
    expect_calls: Sequence[Tuple[Any, str]] = (),
    routers: Optional[Routers] = None,
    **judge_kwargs,
) -> AlignmentReport:
    """Judge one forward and backward of ``test_fn`` against ``ref_fn`` at ``dtype``, with ``ref_fn`` in float64 as
    the truth. Both take ``inputs`` as keyword arguments; modules are copied, so give ``test_fn`` the reference weights
    first (``load_state_dict``). A parameter that requires a gradient and gets none fails, unless ``unused_params``
    names it. ``expect_calls`` lists ``(owner, name)`` pairs, e.g. ``(fused_module, "fused_rms_norm")``, that the test
    run must call at least once (see ``count_calls``), to assert that the fused path really ran rather than a fallback.
    For a mixture-of-experts layer, ``routers`` selects its routers in both modules, and the report counts routing
    flips (``diagnose_routing``). ``judge_kwargs`` go to ``judge`` (``tol``, ``small_tensor_multiplier``)."""
    truth = run_fwd_bwd(ref_fn, inputs, torch.float64, grad_seed, unused_params, routers, role="truth")
    ref = run_fwd_bwd(ref_fn, inputs, dtype, grad_seed, unused_params, routers, role="ref")
    with count_calls(expect_calls) as calls:
        test = run_fwd_bwd(test_fn, inputs, dtype, grad_seed, unused_params, routers, role="test")
    report = require_gradients(judge(test, ref, truth, **judge_kwargs), test=test, ref=ref, truth=truth)
    report = diagnose_routing(report, test, ref, truth)
    for name, count in calls.items():
        report.entries.append(
            Entry(f"<called>/{name}", count > 0, f"called {count} times" if count else "never called")
        )
    return report


def check_optimizer(
    test_fn: Callable,
    ref_fn: Callable,
    params: Union[torch.nn.Module, Dict[str, torch.Tensor]],
    dtype: torch.dtype,
    steps: int = 1,
    grad_seed: int = 0,
    grads_fn: Optional[Callable] = None,
    **judge_kwargs,
) -> AlignmentReport:
    """Judge the weights after each of ``steps`` steps of the optimizer ``test_fn(params)`` builds against those of the
    reference optimizer ``ref_fn(params)`` at ``dtype``, with ``ref_fn`` in float64 as the truth.

    All three runs start from the same weights (``params``: a module, e.g. a model_zoo model, or a name to tensor
    dict) and get the same gradients at every step (``run_optimizer``), so only the update rule is compared. The
    reference must be an independent implementation that runs in float64, e.g. the official one of a paper; its
    internal precision choices (say, an orthogonalization done in bfloat16) must match the tested one's, as for the
    precision recipes of ``check_training``. Several steps exercise the optimizer state (momentum)."""
    kwargs = dict(steps=steps, grad_seed=grad_seed, grads_fn=grads_fn)
    truth = run_optimizer(ref_fn, params, torch.float64, role="truth", **kwargs)
    ref = run_optimizer(ref_fn, params, dtype, role="ref", **kwargs)
    test = run_optimizer(test_fn, params, dtype, role="test", **kwargs)
    return judge(test, ref, truth, **judge_kwargs)


def dtype_mismatches(test: Trace, ref: Trace) -> List[str]:
    """Dtype and device records present in both runs that differ, as ``"<key>: test <dtypes> / ref <dtypes>"``."""
    test_dtypes, ref_dtypes = test.meta["dtypes"], ref.meta["dtypes"]
    return [
        f"{key}: test {test_dtypes[key]} / ref {ref_dtypes[key]}"
        for key in sorted(set(test_dtypes) & set(ref_dtypes))
        if test_dtypes[key] != ref_dtypes[key]
    ]


def blank_control(
    entry: Union[str, tuple], steps: int = 1, seed: int = 0, slack: float = 1.05, cache_dir: Optional[str] = None
) -> AlignmentReport:
    """Defense 3: check the reference implementation itself before trusting any comparison.

    Uses a configuration in which the plugin changes no arithmetic: one device, ``LowLevelZeroPlugin(stage=1,
    precision="bf16")`` (``TorchDDPPlugin()`` in float32 on CPU, where LowLevelZero cannot run), the same AdamW on
    both sides. Every tensor's error must then be within ``slack`` times the baseline error.
    """
    if get_accelerator().name == "cpu":
        plugin = TorchDDPPlugin()
    else:
        plugin = LowLevelZeroPlugin(stage=1, precision="bf16")
    report = check_training(entry, plugin, steps=steps, seed=seed, cache_dir=cache_dir)
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
