"""Helpers shared by both kinds of run: one forward and backward (``fwd_bwd``) and training steps (``training``)."""

from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
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


def missing_grads(grads: Dict[str, Optional[Tensor]], unused_params: Iterable[str] = ()) -> List[str]:
    """Names in ``grads`` (parameter name to gradient, for the parameters that require one) whose gradient is None,
    except ``unused_params``.

    Recorded by every run as ``meta["missing_grads"]`` and failed by ``judge.require_gradients``: the judges compare
    only the tensors the runs recorded, so a gradient missing from every run would otherwise go unnoticed.
    """
    unused = set(unused_params)
    return sorted(name for name, grad in grads.items() if grad is None and name not in unused)


def not_in_optimizer(trainable: Iterable[Tuple[str, Tensor]], optimizer: torch.optim.Optimizer) -> List[str]:
    """Names of the ``(name, tensor)`` pairs that ``optimizer`` does not hold, by identity; ``trainable`` lists what
    should train (for a reference with master weights, their float32 copies).

    Recorded by every training run as ``meta["not_in_optimizer"]`` and failed by ``judge.require_gradients``: an
    ``optimizer_fn`` that builds its own parameter groups (say, no weight decay for norms) and leaves a parameter out
    does so on both sides, so the comparison alone would not notice that the parameter never trains.
    """
    held = {id(p) for group in optimizer.param_groups for p in group["params"]}
    return sorted(name for name, p in trainable if id(p) not in held)


def tie_problems(module: nn.Module, tied: Iterable[Iterable[str]], rename=None) -> Dict[str, str]:
    """How the parameters ``module`` shares between names differ from the declared groups ``tied``.

    Keyed ``tied/<names>``: a declared group that is not one ``Parameter``, or names sharing one without a declaration.
    ``rename`` maps ``module``'s names to those of the model under test (``None`` drops a name). Identity is checked
    within ``module`` only: different runs hold different copies. Recorded as ``meta["tie_problems"]`` and failed by
    ``judge.require_gradients``.
    """
    shared: Dict[int, List[str]] = {}
    for name, p in module.named_parameters(remove_duplicate=False):
        name = rename(name) if rename is not None else name
        if name is not None:
            shared.setdefault(id(p), []).append(name)
    actual = {tuple(sorted(set(names))) for names in shared.values() if len(set(names)) > 1}
    declared = {tuple(sorted(set(group))) for group in tied}
    problems = {f"tied/{','.join(g)}": "declared tied but not one Parameter" for g in declared - actual}
    for group in actual - declared:
        problems[f"tied/{','.join(group)}"] = "one Parameter but not declared in NumericsSpec.tied_params"
    return problems


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


@contextmanager
def count_calls(targets: Sequence[Tuple[Any, str]]) -> Iterator[Dict[str, int]]:
    """Count the calls to each ``getattr(owner, name)`` while the block runs, keyed ``"<owner>.<name>"``.

    Used to assert that a fused kernel really ran. As with ``unittest.mock.patch``, ``owner`` must be where the caller
    looks the name up: the class for a method, the module the calling code references for a function (a name imported
    with ``from module import fn`` before the block is not affected).
    """
    counts, originals = {}, []
    for owner, name in targets:
        key = f"{getattr(owner, '__name__', type(owner).__name__)}.{name}"
        original = getattr(owner, name)
        counts[key] = 0

        def counted(*args, __original=original, __key=key, **kwargs):
            counts[__key] += 1
            return __original(*args, **kwargs)

        # restore exactly what was there: the owner's own attribute, or nothing if it was inherited
        originals.append((owner, name, vars(owner).get(name)))
        setattr(owner, name, counted)
    try:
        yield counts
    finally:
        for owner, name, own in reversed(originals):
            if own is None:
                delattr(owner, name)
            else:
                setattr(owner, name, own)


class RoutingRecorder:
    """Records the experts chosen at every router call of ``module`` while ``active``, for ``judge.diagnose_routing``.

    ``routers`` is a ``spec.Routers`` (or None, which records nothing). Each call appends ``{"module": name,
    "experts": [[e, ...], ...]}``: per token, the indices of the ``top_k`` highest scores, sorted, so that a token whose
    experts only change order is not a flip. Runs record their first forward only and set ``active = False`` after it.
    """

    def __init__(self, module: nn.Module, routers: Any = None):
        self.calls: List[Dict[str, Any]] = []
        self.active = routers is not None
        self.handles = []
        if routers is not None:
            for name, sub in module.named_modules():
                if routers.select(name, sub):
                    self.handles.append(sub.register_forward_hook(partial(self._record, name, routers)))

    def _record(self, name, routers, module, args, output):
        if not self.active:
            return
        scores = routers.scores_fn(output).detach()
        # half precision is widened exactly; float64 is kept, so near ties are not reordered
        scores = scores.to(torch.promote_types(scores.dtype, torch.float32))
        experts = scores.topk(routers.top_k, dim=-1).indices.sort(dim=-1).values
        self.calls.append({"module": name, "experts": experts.reshape(-1, routers.top_k).cpu().tolist()})

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
