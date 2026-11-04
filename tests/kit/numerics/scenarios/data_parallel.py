"""Scenario ②: N-way data parallelism is equivalent to one device accumulating the same N micro-batches.

Both sides run the same plugin with the same precision, so no reference recipe is involved. The single-device run and
the N-device run are separate process groups: each is spawned, saves its ``Trace`` and the two are compared offline
with Megatron-LM's relative bound (``judge_relative_bound``), with Megatron's constants: layer 0 of 32 and the
bfloat16 epsilon whatever the training precision, about 8.3% relative error per tensor. Data-parallel bugs (a
parameter left unsynchronized, a wrong averaging factor) move a gradient by 50% or more.

As in Megatron-LM only the gradients of one step are judged; the traces also hold the weights after the step and the
micro-batch losses, which are not:

- After an Adam step the weights are not a stable target: the first update is about ``lr * sign(grad)``, so a rounding
  difference in a near-zero gradient flips the update of that element. GPT-2 in float32, 4 micro-batches on 2 ranks:
  every gradient within 4e-7 relative error, a bias after the step 4.6e-4.
- The loss here is a random projection of the outputs whose terms largely cancel, so its relative error is inflated.
"""

import os
import tempfile
from typing import Callable, Optional, Tuple, Union

import torch

import colossalai
from colossalai.booster.mixed_precision import MixedPrecision
from colossalai.testing import spawn

from ..capture.training import default_optimizer, prepare, run_data_parallel
from ..judge import AlignmentReport, judge_relative_bound, require_gradients
from ..trace import Trace


def traces(
    entry: Union[str, tuple],
    plugin_fn: Callable,
    mixed_precision: Union[str, MixedPrecision, None] = None,
    world_size: int = 2,
    micro_batches: Optional[int] = None,
    seed: int = 0,
    grad_seed: int = 0,
    optimizer_fn: Callable = default_optimizer,
    directory: Optional[str] = None,
    token_weighted: bool = False,
) -> Tuple[Trace, Trace]:
    """Run ``entry`` on ``world_size`` devices and on one, return ``(test, ref)``.

    Call from the main process: it spawns the two process groups itself. ``plugin_fn`` builds the plugin inside each
    process (e.g. ``functools.partial(LowLevelZeroPlugin, stage=2, precision="bf16")``); it and ``entry`` must be
    picklable, so pass a model_zoo name or an entry made of module-level functions. ``micro_batches`` defaults to
    ``world_size``. The traces are written to ``directory`` (a temporary one by default) as ``ref.safetensors`` and
    ``test.safetensors``.

    By default the loss is the tool's own (a random projection of the outputs, equal micro-batches), so only the
    plugin's gradient reduction is tested. ``token_weighted=True`` uses the entry's ``NumericsSpec.token_loss``
    instead, normalized by the valid tokens of all micro-batches (``TokenLoss``): with padded batches whose valid token
    counts differ, the N-device gradient must still equal the single-device mean over every valid token.
    """
    micro_batches = micro_batches or world_size
    kwargs = dict(
        entry=entry,
        plugin_fn=plugin_fn,
        mixed_precision=mixed_precision,
        micro_batches=micro_batches,
        seed=seed,
        grad_seed=grad_seed,
        optimizer_fn=optimizer_fn,
        token_weighted=token_weighted,
    )
    with tempfile.TemporaryDirectory() if directory is None else _keep(directory) as out:
        ref_path, test_path = os.path.join(out, "ref.safetensors"), os.path.join(out, "test.safetensors")
        spawn(_run, 1, path=ref_path, role="ref", **kwargs)
        spawn(_run, world_size, path=test_path, role="test", **kwargs)
        return Trace.load(test_path), Trace.load(ref_path)


def check(
    *args, bound_dtype: torch.dtype = torch.bfloat16, k: float = 4.0, C: float = 1.03, depth: int = 33, **kwargs
) -> AlignmentReport:
    """``judge_relative_bound`` on the gradients of ``traces(...)``, with Megatron-LM's constants by default; trainable
    parameters left without a gradient fail unless the entry's ``NumericsSpec.unused_params`` names them."""
    test, ref = traces(*args, **kwargs)
    report = judge_relative_bound(_gradients(test), _gradients(ref), bound_dtype, k=k, C=C, depth=depth)
    return require_gradients(report, test=test, ref=ref)


def _gradients(trace: Trace) -> Trace:
    return Trace({k: v for k, v in trace.items() if k.startswith("grad/")}, trace.meta)


def _run(
    rank,
    world_size,
    port,
    entry,
    plugin_fn,
    mixed_precision,
    micro_batches,
    seed,
    grad_seed,
    optimizer_fn,
    token_weighted,
    path,
    role,
):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    s = prepare(entry, micro_batches, seed)
    trace = run_data_parallel(
        s, plugin_fn(), micro_batches, mixed_precision, optimizer_fn, grad_seed, token_weighted, role=role
    )
    if rank == 0:
        trace.save(path)


class _keep:
    def __init__(self, directory: str):
        self.directory = directory

    def __enter__(self) -> str:
        os.makedirs(self.directory, exist_ok=True)
        return self.directory

    def __exit__(self, *exc) -> None:
        pass
