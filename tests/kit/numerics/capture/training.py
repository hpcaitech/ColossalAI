"""Training steps of a model_zoo entry, through a booster plugin or as the plain PyTorch reference.

Each function runs once and returns one ``Trace``; ``scenarios`` decide which runs to make and how to compare them.
Depends on ColossalAI, so ``capture/__init__`` does not import it."""

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
from torch import Tensor, nn
from torch.utils._pytree import tree_flatten, tree_map

from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.mixed_precision import MixedPrecision
from colossalai.booster.mixed_precision.fp16_torch import TorchAMPOptimizer
from colossalai.booster.plugin import Plugin

from ..compare import flatten_tensors
from ..spec import NumericsSpec
from ..trace import Trace
from .recipes import Recipe, recipe_for
from .utils import cast_floating, full_precision_matmul, upstream_loss


def default_optimizer(params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=1e-3)


@dataclass
class Prepared:
    """One entry ready to run: weights built once, batches drawn once, shared by every run."""

    name: str
    model: nn.Module
    reference: nn.Module
    batches: List[Dict[str, Any]]
    select: Callable[[Any], Any]
    param_map: Optional[Callable[[str], Optional[str]]]
    seed: int


def prepare(entry: Union[str, tuple], steps: int = 1, seed: int = 0) -> Prepared:
    """Build the model, its reference and ``steps`` batches for a model_zoo entry (a name or an entry tuple).

    Models stay in train mode with every dropout probability set to 0, so training-only code paths run but nothing
    is random.
    """
    name = entry if isinstance(entry, str) else "custom"
    if isinstance(entry, str):
        from tests.kit.model_zoo import model_zoo

        entry = model_zoo[entry]
    model_fn, data_gen_fn, output_transform_fn, _, attribute = entry
    spec: Optional[NumericsSpec] = attribute.numerics if attribute is not None else None
    if spec is None:
        raise ValueError(f"{name} has no NumericsSpec; add ModelAttribute(numerics=NumericsSpec()) to its entry")

    device = get_accelerator().get_current_device()
    torch.manual_seed(seed)
    model = _disable_dropout(model_fn()).train().to(device)
    reference = model
    if spec.reference_fn is not None:
        reference = _disable_dropout(spec.reference_fn()).train().to(device)
        _copy_weights(reference, model, spec.param_map)
    batches = []
    for step in range(steps):
        torch.manual_seed(seed + 1 + step)
        batches.append(tree_map(lambda x: x.to(device) if isinstance(x, Tensor) else x, data_gen_fn()))

    def select(output: Any) -> Any:
        output = output_transform_fn(output)
        return spec.output_fn(output) if spec.output_fn is not None else output

    return Prepared(name, model, reference, batches, select, spec.param_map, seed)


def run_reference(
    s: Prepared, recipe: Recipe, optimizer_fn: Callable = default_optimizer, grad_seed: int = 0, **meta
) -> Trace:
    """Train the reference for ``len(s.batches)`` steps following ``recipe``, in plain PyTorch.

    Records ``out/...``, input gradients ``grad/<input>`` and parameter gradients ``grad/param/<name>`` of the first
    step, the weights after it as ``param/<name>``, and ``loss/<step>`` for every step. Parameter names are those of the
    model under test (``param_map`` is applied). The dtypes seen on the way are recorded in ``meta["dtypes"]``.
    """
    device = get_accelerator().get_current_device()
    module = copy.deepcopy(s.reference)
    if not recipe.autocast:
        module.to(recipe.dtype)
    named = list(module.named_parameters())
    if recipe.master_weights:
        master = [nn.Parameter(p.detach().float()) for _, p in named]
        optimizer = optimizer_fn(master)
    else:
        optimizer = optimizer_fn([p for _, p in named])
    autocast = partial(torch.autocast, device.type, dtype=recipe.dtype) if recipe.autocast else nullcontext
    audit = _DtypeAudit(module, s.param_map)

    torch.manual_seed(s.seed)
    results = {}
    with full_precision_matmul():
        for step, batch in enumerate(s.batches):
            inputs = _leaves(batch, None if recipe.autocast else recipe.dtype)
            with autocast():
                raw = module(**inputs)
            outputs = flatten_tensors(s.select(raw), "out")
            loss = _loss(outputs, grad_seed + step)
            (loss * recipe.loss_scale if recipe.loss_scale else loss).backward()
            if recipe.loss_scale:
                for _, p in named:
                    if p.grad is not None:
                        p.grad.mul_(1.0 / recipe.loss_scale)
            if step == 0:
                results.update(_first_step(outputs, inputs, {n: p.grad for n, p in named}, s.param_map))
            if recipe.master_weights:
                for (_, p), m in zip(named, master):
                    m.grad = None if p.grad is None else p.grad.float()
            optimizer.step()
            if recipe.master_weights:
                with torch.no_grad():
                    for (_, p), m in zip(named, master):
                        p.copy_(m.to(p.dtype))
            if step == 0:
                results.update(_params(named, s.param_map))
                audit.record_optimizer(optimizer)
            optimizer.zero_grad(set_to_none=True)
            module.zero_grad(set_to_none=True)
            results[f"loss/{step}"] = loss.detach()
    audit.remove()
    return Trace.capture(results, dtypes=audit.records, recipe=_describe(recipe), **_meta(s, grad_seed, meta))


def run_boosted(
    s: Prepared,
    plugin: Plugin,
    mixed_precision: Union[str, MixedPrecision, None] = None,
    optimizer_fn: Callable = default_optimizer,
    grad_seed: int = 0,
    **meta,
) -> Trace:
    """Train the model under test for ``len(s.batches)`` steps through ``Booster(plugin, mixed_precision)``.

    Records the same names as ``run_reference``. Inputs are passed in float32; casting them is the plugin's job.
    """
    recipe = recipe_for(plugin, mixed_precision)
    input_dtype = None if recipe.autocast else recipe.dtype
    module = copy.deepcopy(s.model)
    optimizer = optimizer_fn(module.parameters())
    booster = Booster(plugin=plugin, mixed_precision=mixed_precision)
    module, optimizer, *_ = booster.boost(module, optimizer)
    module.train()
    inner = module.unwrap()
    audit = _DtypeAudit(inner)
    # ZeRO keeps reduced gradients in its own store and clears param.grad; get_param_grad gathers them back.
    get_grad = getattr(optimizer, "get_param_grad", lambda p: p.grad)

    torch.manual_seed(s.seed)
    results = {}
    with full_precision_matmul():
        for step, batch in enumerate(s.batches):
            inputs = _leaves(batch, None)
            outputs = flatten_tensors(s.select(module(**inputs)), "out")
            loss = _loss(outputs, grad_seed + step)
            booster.backward(loss, optimizer)
            if isinstance(optimizer, TorchAMPOptimizer):
                scale = optimizer.scaler.get_scale()
                optimizer.unscale_grad()
            named = list(inner.named_parameters())
            if step == 0:
                results.update(_first_step(outputs, inputs, {n: get_grad(p) for n, p in named}, None))
                # inputs went in as float32 and the plugin cast them; report their gradients in the dtype it computed
                # in (exact: the values come from that dtype), as the reference does
                for key in inputs:
                    if f"grad/{key}" in results and input_dtype is not None:
                        results[f"grad/{key}"] = results[f"grad/{key}"].to(input_dtype)
            optimizer.step()
            if isinstance(optimizer, TorchAMPOptimizer) and optimizer.scaler.get_scale() < scale:
                raise RuntimeError(
                    f"step {step}: fp16 gradients overflowed at loss scale {scale}, the step was skipped"
                )
            if step == 0:
                results.update(_params(named, None))
                audit.record_optimizer(_unwrap_optimizer(optimizer))
            optimizer.zero_grad()
            results[f"loss/{step}"] = loss.detach()
    audit.remove()
    meta = dict(meta, plugin=type(plugin).__name__, mixed_precision=str(mixed_precision))
    return Trace.capture(results, dtypes=audit.records, **_meta(s, grad_seed, meta))


def _unwrap_optimizer(optimizer) -> torch.optim.Optimizer:
    # plugins and mixed precision may each wrap the torch optimizer
    while hasattr(optimizer, "optim"):
        optimizer = optimizer.optim
    return optimizer


def _loss(outputs: Dict[str, Tensor], grad_seed: int) -> Tensor:
    # upstream_loss divided by the number of output elements: mean-reduced like a real training loss, so that fp16 loss
    # scaling (2**16 by default) does not overflow. Computed in at least float32, as real losses are; otherwise the
    # loss scale itself, flowing back into a float16 loss, is already inf.
    outputs = {k: v.to(torch.promote_types(v.dtype, torch.float32)) for k, v in outputs.items()}
    return upstream_loss(outputs, grad_seed) / sum(out.numel() for out in outputs.values() if out.requires_grad)


def _meta(s: Prepared, grad_seed: int, meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(model=s.name, seed=s.seed, grad_seed=grad_seed, steps=len(s.batches), **meta)


def _describe(recipe: Recipe) -> str:
    parts = [str(recipe.dtype).replace("torch.", ""), "autocast" if recipe.autocast else "cast"]
    if recipe.master_weights:
        parts.append("float32 master weights")
    if recipe.loss_scale:
        parts.append(f"loss scale {recipe.loss_scale:g}")
    return ", ".join(parts)


def _leaves(batch: Dict[str, Any], dtype: Optional[torch.dtype]) -> Dict[str, Any]:
    leaves = {}
    for key, value in (cast_floating(batch, dtype) if dtype is not None else batch).items():
        if isinstance(value, Tensor) and value.is_floating_point():
            value = value.detach().clone().requires_grad_(True)
        leaves[key] = value
    return leaves


def _rename(name: str, param_map) -> Optional[str]:
    return param_map(name) if param_map is not None else name


def _first_step(outputs, inputs, grads, param_map) -> Dict[str, Tensor]:
    results = {key: out.detach() for key, out in outputs.items()}
    for key, value in inputs.items():
        if isinstance(value, Tensor) and value.grad is not None:
            results[f"grad/{key}"] = value.grad.detach().clone()
    for name, grad in grads.items():
        name = _rename(name, param_map)
        if name is not None and grad is not None:
            results[f"grad/param/{name}"] = grad.detach().clone()
    return results


def _params(named, param_map) -> Dict[str, Tensor]:
    results = {}
    for name, p in named:
        name = _rename(name, param_map)
        if name is not None:
            results[f"param/{name}"] = p.detach().clone()
    return results


def _disable_dropout(model: nn.Module) -> nn.Module:
    # nn.Dropout modules and the float probabilities HuggingFace keeps on modules for functional dropout
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.p = 0.0
        for key, value in vars(module).items():
            if "dropout" in key and isinstance(value, float):
                setattr(module, key, 0.0)
    return model


def _copy_weights(reference: nn.Module, model: nn.Module, param_map) -> None:
    source = dict(model.named_parameters())
    with torch.no_grad():
        for name, param in reference.named_parameters():
            mapped = _rename(name, param_map)
            if mapped is None:
                continue
            if mapped not in source:
                raise KeyError(f"reference parameter {name} maps to {mapped}, which the model under test does not have")
            param.copy_(source[mapped])


def _dtypes(obj: Any) -> str:
    leaves, _ = tree_flatten(obj)
    return ",".join(str(x.dtype).replace("torch.", "") for x in leaves if isinstance(x, Tensor)) or "-"


class _DtypeAudit:
    """Records the dtypes a run actually uses, to check that test and reference follow the same recipe.

    Per module (first call): input, output and own-weight dtypes, keyed ``module/<name>``. Module names are only
    comparable between identical architectures, so a reference with ``param_map`` records parameters only. Per
    parameter: ``param/<name>`` and ``grad/<name>``. For the optimizer: ``optim/<state key>`` with the dtypes found.
    """

    def __init__(self, module: nn.Module, param_map=None):
        self.records: Dict[str, str] = {}
        self.handles = []
        if param_map is None:
            for name, sub in module.named_modules():
                hook = partial(self._forward, f"module/{name or '<root>'}")
                self.handles.append(sub.register_forward_hook(hook, with_kwargs=True))
        for name, p in module.named_parameters():
            name = _rename(name, param_map)
            if name is not None:
                self.records[f"param/{name}"] = _dtypes(p)
                self.handles.append(p.register_hook(partial(self._grad, f"grad/{name}")))

    def _forward(self, key, module, args, kwargs, output):
        if key not in self.records:
            weights = list(module.parameters(recurse=False))
            self.records[key] = f"in={_dtypes((args, kwargs))} out={_dtypes(output)} w={_dtypes(weights)}"

    def _grad(self, key, grad):
        self.records.setdefault(key, _dtypes(grad))

    def record_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        found: Dict[str, set] = {}
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    found.setdefault(f"optim/{key}", set()).add(_dtypes(value))
        self.records.update({key: ",".join(sorted(v)) for key, v in found.items()})

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
