"""Training steps of a model_zoo entry, through a booster plugin or as the plain PyTorch reference.

Each function runs once and returns one ``Trace``; ``scenarios`` decide which runs to make and how to compare them.
Depends on ColossalAI, so ``capture/__init__`` does not import it."""

import copy
import os
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils._pytree import tree_flatten, tree_map

from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.mixed_precision import MixedPrecision
from colossalai.booster.mixed_precision.fp16_torch import TorchAMPOptimizer
from colossalai.booster.plugin import Plugin

from ..compare import flatten_tensors
from ..spec import NumericsSpec, Routers
from ..trace import Trace
from .recipes import Recipe, recipe_for
from .utils import RoutingRecorder, cast_floating, full_precision_matmul, missing_grads, upstream_loss


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
    unused_params: Tuple[str, ...] = ()
    routers: Optional[Routers] = None


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

    return Prepared(name, model, reference, batches, select, spec.param_map, seed, spec.unused_params, spec.routers)


def run_reference(
    s: Prepared, recipe: Recipe, optimizer_fn: Callable = default_optimizer, grad_seed: int = 0, **meta
) -> Trace:
    """Train the reference for ``len(s.batches)`` steps following ``recipe``, in plain PyTorch.

    Records ``out/...``, input gradients ``grad/<input>`` and parameter gradients ``grad/param/<name>`` of the first
    step, the weights after it as ``param/<name>``, and ``loss/<step>`` for every step. Parameter names are those of the
    model under test (``param_map`` is applied). The dtypes seen on the way are recorded in ``meta["dtypes"]``, the
    trainable parameters left without a gradient in ``meta["missing_grads"]`` and, if the entry declares ``routers``,
    the experts of the first forward in ``meta["routing"]``.
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
    routing = RoutingRecorder(module, s.routers)

    torch.manual_seed(s.seed)
    results = {}
    with full_precision_matmul():
        for step, batch in enumerate(s.batches):
            inputs = _leaves(batch, None if recipe.autocast else recipe.dtype)
            with autocast():
                raw = module(**inputs)
            routing.active = False
            outputs = flatten_tensors(s.select(raw), "out")
            loss = _loss(outputs, grad_seed + step)
            (loss * recipe.loss_scale if recipe.loss_scale else loss).backward()
            if recipe.loss_scale:
                for _, p in named:
                    if p.grad is not None:
                        p.grad.mul_(1.0 / recipe.loss_scale)
            if step == 0:
                results.update(_first_step(outputs, inputs, {n: p.grad for n, p in named}, s.param_map))
                missing = _missing_grads(named, lambda p: p.grad, s)
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
    routing.remove()
    return Trace.capture(
        results,
        dtypes=audit.records,
        recipe=_describe(recipe),
        missing_grads=missing,
        **_routing_meta(s, routing),
        **_meta(s, grad_seed, meta),
    )


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
    routing = RoutingRecorder(inner, s.routers)
    # ZeRO keeps reduced gradients in its own store and clears param.grad; get_param_grad gathers them back.
    get_grad = getattr(optimizer, "get_param_grad", lambda p: p.grad)

    torch.manual_seed(s.seed)
    results = {}
    with full_precision_matmul():
        for step, batch in enumerate(s.batches):
            inputs = _leaves(batch, None)
            outputs = flatten_tensors(s.select(module(**inputs)), "out")
            routing.active = False
            loss = _loss(outputs, grad_seed + step)
            booster.backward(loss, optimizer)
            if isinstance(optimizer, TorchAMPOptimizer):
                scale = optimizer.scaler.get_scale()
                optimizer.unscale_grad()
            named = list(inner.named_parameters())
            if step == 0:
                results.update(_first_step(outputs, inputs, {n: get_grad(p) for n, p in named}, None))
                missing = _missing_grads(named, get_grad, s, mapped=False)
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
    routing.remove()
    meta = dict(meta, plugin=type(plugin).__name__, mixed_precision=str(mixed_precision))
    return Trace.capture(
        results, dtypes=audit.records, missing_grads=missing, **_routing_meta(s, routing), **_meta(s, grad_seed, meta)
    )


def run_data_parallel(
    s: Prepared,
    plugin: Plugin,
    micro_batches: int,
    mixed_precision: Union[str, MixedPrecision, None] = None,
    optimizer_fn: Callable = default_optimizer,
    grad_seed: int = 0,
    **meta,
) -> Trace:
    """Train through the plugin with ``micro_batches`` micro-batches per step, spread over the data-parallel ranks.

    With world size W, rank r runs micro-batches r, r + W, ... and accumulates their gradients; the plugin then
    averages over the ranks. Each micro-batch loss is scaled by W / micro_batches, so every world size computes the
    gradient of the mean loss over all micro-batches. ``s`` must hold ``steps * micro_batches`` batches; micro-batch i
    of step t is ``s.batches[t * micro_batches + i]`` with upstream gradient seed ``grad_seed + t * micro_batches + i``.

    Records ``grad/param/<name>`` (the averaged gradient of the first step), ``param/<name>`` (the weights after it)
    and ``loss/<step>/<micro-batch>``, and in ``meta["missing_grads"]`` the trainable parameters without a gradient.
    The trace is the same on every rank.
    """
    world_size, rank = dist.get_world_size(), dist.get_rank()
    if micro_batches % world_size or len(s.batches) % micro_batches:
        raise ValueError(
            f"{micro_batches} micro-batches cannot be split over {world_size} ranks and {len(s.batches)} batches"
        )
    recipe = recipe_for(plugin, mixed_precision)
    module = copy.deepcopy(s.model)
    optimizer = optimizer_fn(module.parameters())
    booster = Booster(plugin=plugin, mixed_precision=mixed_precision)
    module, optimizer, *_ = booster.boost(module, optimizer)
    module.train()
    inner = module.unwrap()
    get_grad = getattr(optimizer, "get_param_grad", lambda p: p.grad)

    torch.manual_seed(s.seed)
    results = {}
    with full_precision_matmul():
        for step in range(len(s.batches) // micro_batches):
            losses = {}
            for micro in range(rank, micro_batches, world_size):
                index = step * micro_batches + micro
                outputs = flatten_tensors(s.select(module(**_leaves(s.batches[index], None))), "out")
                loss = _loss(outputs, grad_seed + index)
                booster.backward(loss * (world_size / micro_batches), optimizer)
                losses[f"loss/{step}/{micro}"] = loss.detach().float().cpu()
            if isinstance(optimizer, TorchAMPOptimizer):
                scale = optimizer.scaler.get_scale()
                optimizer.unscale_grad()
            named = list(inner.named_parameters())
            if step == 0:
                # get_param_grad is collective: every rank gathers every parameter
                grads = {f"grad/param/{n}": get_grad(p) for n, p in named}
                results.update({k: g.detach().clone() for k, g in grads.items() if g is not None})
                missing = missing_grads(
                    {n: grads[f"grad/param/{n}"] for n, p in named if p.requires_grad}, s.unused_params
                )
            optimizer.step()
            if isinstance(optimizer, TorchAMPOptimizer) and optimizer.scaler.get_scale() < scale:
                raise RuntimeError(
                    f"step {step}: fp16 gradients overflowed at loss scale {scale}, the step was skipped"
                )
            if step == 0:
                results.update(_params(named, None))
            optimizer.zero_grad()
            gathered = [None] * world_size
            dist.all_gather_object(gathered, losses)
            for part in gathered:
                results.update(part)
    steps = len(s.batches) // micro_batches
    meta = dict(
        meta, plugin=type(plugin).__name__, mixed_precision=str(mixed_precision), world_size=world_size, steps=steps
    )
    return Trace.capture(
        results,
        dtype=str(recipe.dtype).replace("torch.", ""),
        micro_batches=micro_batches,
        missing_grads=missing,
        **_meta(s, grad_seed, meta),
    )


CHECKPOINT_PARTS = ("model", "optimizer", "lr_scheduler")


def run_steps(
    s: Prepared,
    plugin: Plugin,
    start: int,
    stop: int,
    mixed_precision: Union[str, MixedPrecision, None] = None,
    optimizer_fn: Callable = default_optimizer,
    grad_seed: int = 0,
    load: Optional[str] = None,
    save: Optional[str] = None,
    parts: Iterable[str] = CHECKPOINT_PARTS,
    shard_optimizer: bool = True,
    **meta,
) -> Trace:
    """Train steps ``start`` to ``stop - 1`` of a ``len(s.batches)``-step run through the plugin, one batch per step.

    The learning rate warms up linearly over the whole run (``LinearLR``), so it differs at every step. With ``load``,
    the ``parts`` of a checkpoint written by ``save`` are loaded first (with the booster's own ``load_*`` methods);
    with ``save``, they are written after the last step, the model sharded in safetensors with its HuggingFace config,
    the optimizer sharded unless ``shard_optimizer=False``.

    Records ``loss/<step>`` for every step run and ``param/<name>`` after the last one. Each rank runs the same batch;
    data parallelism itself is checked by ``run_data_parallel``.
    """
    module = copy.deepcopy(s.model)
    optimizer = optimizer_fn(module.parameters())
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.25, total_iters=len(s.batches))
    booster = Booster(plugin=plugin, mixed_precision=mixed_precision)
    module, optimizer, _, _, scheduler = booster.boost(module, optimizer, lr_scheduler=scheduler)
    module.train()
    parts = set(parts)
    if load is not None:
        if "model" in parts:
            booster.load_model(module, os.path.join(load, "model"))
        if "optimizer" in parts:
            booster.load_optimizer(optimizer, os.path.join(load, "optimizer"))
        if "lr_scheduler" in parts:
            booster.load_lr_scheduler(scheduler, os.path.join(load, "lr_scheduler.pt"))

    torch.manual_seed(s.seed)
    results = {}
    with full_precision_matmul():
        for step in range(start, stop):
            outputs = flatten_tensors(s.select(module(**_leaves(s.batches[step], None))), "out")
            loss = _loss(outputs, grad_seed + step)
            booster.backward(loss, optimizer)
            optimizer.step()
            if isinstance(optimizer, TorchAMPOptimizer) and optimizer.scaler.get_scale() != _initial_scale(optimizer):
                raise RuntimeError(f"step {step}: the fp16 loss scale changed, so the run is not reproducible")
            optimizer.zero_grad()
            scheduler.step()
            results[f"loss/{step}"] = loss.detach().float().cpu()
    results.update(_params(list(module.unwrap().named_parameters()), None))
    if save is not None:
        booster.save_model(module, os.path.join(save, "model"), shard=True, use_safetensors=True)
        booster.save_optimizer(optimizer, os.path.join(save, "optimizer"), shard=shard_optimizer)
        booster.save_lr_scheduler(scheduler, os.path.join(save, "lr_scheduler.pt"))
    meta = dict(meta, plugin=type(plugin).__name__, mixed_precision=str(mixed_precision), start=start, stop=stop)
    return Trace.capture(results, **_meta(s, grad_seed, meta))


def _initial_scale(optimizer: TorchAMPOptimizer) -> float:
    return optimizer.scaler._init_scale


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


def _routing_meta(s: Prepared, routing: RoutingRecorder) -> Dict[str, Any]:
    return {"routing": routing.calls} if s.routers is not None else {}


def _meta(s: Prepared, grad_seed: int, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {**dict(model=s.name, seed=s.seed, grad_seed=grad_seed, steps=len(s.batches)), **meta}


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


def _missing_grads(named, get_grad, s: Prepared, mapped: bool = True) -> List[str]:
    # names of the model under test: the reference's are mapped with param_map, the boosted model's already are
    grads = {}
    for name, p in named:
        name = _rename(name, s.param_map) if mapped else name
        if name is not None and p.requires_grad:
            grads[name] = get_grad(p)
    return missing_grads(grads, s.unused_params)


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


def _dtypes(obj: Any, ordered: bool = True) -> str:
    leaves, _ = tree_flatten(obj)
    names = [str(x.dtype).replace("torch.", "") for x in leaves if isinstance(x, Tensor)]
    return ",".join(names if ordered else sorted(names)) or "-"


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
            # inputs unordered: a rewritten forward (e.g. shardformer's) passes the same tensors in another order
            inputs = _dtypes((args, kwargs), ordered=False)
            self.records[key] = f"in={inputs} out={_dtypes(output)} w={_dtypes(weights)}"

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
