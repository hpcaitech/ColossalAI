"""Training steps of a model_zoo entry, through a booster plugin or as the plain PyTorch reference.

Each function runs once and returns one ``Trace``; ``scenarios`` decide which runs to make and how to compare them.
Depends on ColossalAI, so ``capture/__init__`` does not import it."""

import copy
import inspect
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
from colossalai.booster.plugin import GeminiPlugin, Plugin
from colossalai.booster.plugin.torch_fsdp_plugin import TorchFSDPPlugin

from ..compare import flatten_tensors
from ..spec import NumericsSpec, Routers
from ..trace import Trace
from .plugins import (
    FirstMomentIsGradient,
    exposes_gradients,
    full_gradients,
    full_parameters,
    inner_module,
    loss_scale,
    make_optimizer,
    unwrap_optimizer,
)
from .recipes import Recipe, UnsupportedRecipe, recipe_for
from .utils import (
    RoutingRecorder,
    cast_floating,
    full_precision_matmul,
    missing_grads,
    not_in_optimizer,
    tie_problems,
    upstream_loss,
)


def default_optimizer(params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=1e-3)


@dataclass
class Prepared:
    """One entry ready to run: weights built once, batches drawn once, shared by every run.

    ``model`` and ``reference`` stay on the CPU; each run copies the one it needs to the device and drops it when done,
    so only one model is on the device at a time (in float64 for the truth: cast on the way, never held in float32
    there too)."""

    name: str
    model: nn.Module
    reference: nn.Module
    batches: List[Dict[str, Any]]
    select: Callable[[Any], Any]
    param_map: Optional[Callable[[str], Optional[str]]]
    seed: int
    unused_params: Tuple[str, ...] = ()
    routers: Optional[Routers] = None
    token_loss: Optional[Callable] = None
    code: str = ""
    tied_params: Tuple[Tuple[str, ...], ...] = ()


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
    model = _build(model_fn, spec.prepare_fn).train()
    reference = model
    if spec.reference_fn is not None:
        reference = _build(spec.reference_fn, spec.prepare_fn).train()
        _copy_weights(reference, model, spec.param_map)
    batches = []
    for step in range(steps):
        torch.manual_seed(seed + 1 + step)
        batches.append(tree_map(lambda x: x.to(device) if isinstance(x, Tensor) else x, data_gen_fn()))

    def select(output: Any) -> Any:
        output = output_transform_fn(output)
        return spec.output_fn(output) if spec.output_fn is not None else output

    code = _code(output_transform_fn, spec.output_fn, spec.param_map)
    if spec.routers is not None:
        code += _code(spec.routers.select, spec.routers.scores_fn) + f"top_k={spec.routers.top_k}"
    return Prepared(
        name,
        model,
        reference,
        batches,
        select,
        spec.param_map,
        seed,
        spec.unused_params,
        spec.routers,
        spec.token_loss,
        code,
        spec.tied_params,
    )


def run_reference(
    s: Prepared, recipe: Recipe, optimizer_fn: Callable = default_optimizer, grad_seed: int = 0, **meta
) -> Trace:
    """Train the reference for ``len(s.batches)`` steps following ``recipe``, in plain PyTorch.

    Records ``out/...``, input gradients ``grad/<input>`` and parameter gradients ``grad/param/<name>`` of the first
    step, the weights after it as ``param/<name>``, and ``loss/<step>`` for every step. Parameter names are those of the
    model under test (``param_map`` is applied). The dtypes seen on the way are recorded in ``meta["dtypes"]``, the
    trainable parameters left without a gradient in ``meta["missing_grads"]`` and, if the entry declares ``routers``,
    the experts of the first forward in ``meta["routing"]``. The parameter contract of the entry is checked too
    (``not_in_optimizer``, ``tie_problems``).
    """
    device = get_accelerator().get_current_device()
    module = copy.deepcopy(s.reference).to(device=device, dtype=None if recipe.autocast else recipe.dtype)
    named = list(module.named_parameters())
    if recipe.master_weights:
        # from the weights as cast, or from the float32 ones the model was built with (same names: module is a copy)
        start = dict(s.reference.named_parameters()) if recipe.master_from_float32 else dict(named)
        master = [nn.Parameter(start[n].detach().float().to(recipe.optimizer_device or p.device)) for n, p in named]
        optimizer = optimizer_fn(master)
        trainable = [(n, m) for (n, p), m in zip(named, master) if p.requires_grad]
    else:
        optimizer = optimizer_fn([p for _, p in named])
        trainable = [(n, p) for n, p in named if p.requires_grad]
    contract = _contract(s, module, trainable, optimizer, s.param_map)
    autocast = partial(torch.autocast, device.type, dtype=recipe.dtype) if recipe.autocast else nullcontext
    audit = _RecipeAudit(module, s.param_map)
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
            grads = {n: _unscaled(p.grad, recipe) for n, p in named}
            if step == 0:
                # recorded in the dtype they were computed in, as the plugin's are
                recorded = {n: None if g is None else g.to(p.dtype) for (n, p), g in zip(named, grads.values())}
                results.update(_first_step(outputs, inputs, recorded, s.param_map))
                missing = _missing_grads(named, lambda p: p.grad, s)
            if recipe.master_weights:
                for (n, p), m in zip(named, master):
                    m.grad = None if grads[n] is None else grads[n].float().to(m.device)
            else:
                for n, p in named:
                    p.grad = grads[n]
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
        **contract,
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

    Records the same names as ``run_reference``. Inputs are passed in float32; casting them is the plugin's job. For
    Gemini and FSDP the gradients are read from the optimizer (see ``capture.plugins``): ``optimizer_fn`` must then
    come from ``FirstMomentIsGradient``, on the reference side too.
    """
    recipe = recipe_for(plugin, mixed_precision)
    input_dtype = None if recipe.autocast else recipe.dtype
    working_dtype = torch.float32 if recipe.autocast else recipe.dtype
    module = copy.deepcopy(s.model).to(get_accelerator().get_current_device())
    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    # FSDP replaces the parameters with plain tensors while it runs and adds its flat parameter to the root module:
    # look the weights up by their original names
    weight_names = _weight_names(module) if isinstance(plugin, TorchFSDPPlugin) else None
    optimizer = make_optimizer(plugin, optimizer_fn, [p for p in module.parameters() if p.requires_grad])
    contract = _contract(s, module, _trainable(module), optimizer)
    booster = Booster(plugin=plugin, mixed_precision=mixed_precision)
    module, optimizer, *_ = booster.boost(module, optimizer)
    module.train()
    inner = inner_module(plugin, module)
    _check_ties_after_boost(contract, s, plugin, inner)
    audit = _RecipeAudit(inner, weight_names=weight_names)
    routing = RoutingRecorder(inner, s.routers)
    names = [n for n, _ in s.model.named_parameters()]
    gradients_after_backward = exposes_gradients(plugin)
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
            scale = loss_scale(optimizer)
            if isinstance(optimizer, TorchAMPOptimizer):
                optimizer.unscale_grad()
            if step == 0:
                if gradients_after_backward:
                    named = list(inner.named_parameters())
                    grads = {n: get_grad(p) for n, p in named}
                    missing = _missing_grads(named, get_grad, s, mapped=False)
                else:
                    grads = {}
                results.update(_first_step(_as_dtype(outputs, working_dtype, plugin), inputs, grads, None))
                # inputs went in as float32 and the plugin cast them; report their gradients in the dtype it computed
                # in (exact: the values come from that dtype), as the reference does
                for key in inputs:
                    if f"grad/{key}" in results and input_dtype is not None:
                        results[f"grad/{key}"] = results[f"grad/{key}"].to(input_dtype)
            optimizer.step()
            if scale is not None and loss_scale(optimizer) < scale:
                raise RuntimeError(
                    f"step {step}: fp16 gradients overflowed at loss scale {scale}, the step was skipped"
                )
            if step == 0:
                if not gradients_after_backward:
                    grads = full_gradients(plugin, module, optimizer, trainable)
                    missing = missing_grads(grads, s.unused_params)
                    # exp_avg is float32; the gradient was computed in working_dtype, so the cast back is exact
                    results.update({f"grad/param/{n}": g.to(working_dtype) for n, g in grads.items() if g is not None})
                params = full_parameters(plugin, module, names)
                results.update({f"param/{n}": p.to(working_dtype) for n, p in params.items()})
                audit.record_optimizer(unwrap_optimizer(optimizer))
            optimizer.zero_grad()
            results[f"loss/{step}"] = loss.detach()
    audit.remove()
    routing.remove()
    meta = dict(meta, plugin=type(plugin).__name__, mixed_precision=str(mixed_precision))
    return Trace.capture(
        results,
        dtypes=audit.records,
        missing_grads=missing,
        **contract,
        **_routing_meta(s, routing),
        **_meta(s, grad_seed, meta),
    )


def run_data_parallel(
    s: Prepared,
    plugin: Plugin,
    micro_batches: int,
    mixed_precision: Union[str, MixedPrecision, None] = None,
    optimizer_fn: Callable = default_optimizer,
    grad_seed: int = 0,
    token_weighted: bool = False,
    **meta,
) -> Trace:
    """Train through the plugin with ``micro_batches`` micro-batches per step, spread over the data-parallel ranks.

    With world size W, rank r runs micro-batches r, r + W, ... and accumulates their gradients; the plugin then
    averages over the ranks. Each micro-batch loss is scaled by W / micro_batches, so every world size computes the
    gradient of the mean loss over all micro-batches. ``s`` must hold ``steps * micro_batches`` batches; micro-batch i
    of step t is ``s.batches[t * micro_batches + i]`` with upstream gradient seed ``grad_seed + t * micro_batches + i``.

    With ``token_weighted``, the loss is the entry's ``NumericsSpec.token_loss`` instead: each micro-batch's summed loss
    is scaled by W / (valid tokens of all the step's micro-batches), so every world size computes the gradient of the
    mean over all valid tokens. Every rank holds all batches, so the count needs no communication. A micro-batch
    without valid tokens still runs forward and backward (its gradient is zero, and every rank takes part in the
    reduction) and records its summed loss, 0; a step without any valid token raises ``ValueError`` on every rank
    before its first forward, as its mean is undefined.

    Records ``grad/param/<name>`` (the averaged gradient of the first step), ``param/<name>`` (the weights after it)
    and ``loss/<step>/<micro-batch>``, and in ``meta["missing_grads"]`` the trainable parameters without a gradient.
    The trace is the same on every rank. Gemini and FSDP gradients are read from the optimizer (``capture.plugins``),
    so their optimizer runs with ``beta1 = 0`` on both sides.
    """
    world_size, rank = dist.get_world_size(), dist.get_rank()
    if micro_batches % world_size or len(s.batches) % micro_batches:
        raise ValueError(
            f"{micro_batches} micro-batches cannot be split over {world_size} ranks and {len(s.batches)} batches"
        )
    if token_weighted and s.token_loss is None:
        raise ValueError(f"{s.name}: token_weighted needs NumericsSpec(token_loss=TokenLoss(...))")
    accumulating = micro_batches > world_size
    if accumulating and isinstance(plugin, GeminiPlugin) and not plugin.gemini_config["enable_gradient_accumulation"]:
        raise UnsupportedRecipe(
            "GeminiPlugin overwrites gradients on a second backward; accumulating micro-batches needs "
            "GeminiPlugin(enable_gradient_accumulation=True)"
        )
    recipe = recipe_for(plugin, mixed_precision)
    working_dtype = torch.float32 if recipe.autocast else recipe.dtype
    gradients_after_backward = exposes_gradients(plugin)
    if not gradients_after_backward:
        optimizer_fn = FirstMomentIsGradient(optimizer_fn)
    module = copy.deepcopy(s.model).to(get_accelerator().get_current_device())
    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    optimizer = make_optimizer(plugin, optimizer_fn, [p for p in module.parameters() if p.requires_grad])
    contract = _contract(s, module, _trainable(module), optimizer)
    booster = Booster(plugin=plugin, mixed_precision=mixed_precision)
    module, optimizer, *_ = booster.boost(module, optimizer)
    module.train()
    inner = inner_module(plugin, module)
    _check_ties_after_boost(contract, s, plugin, inner)
    names = [n for n, _ in s.model.named_parameters()]
    get_grad = getattr(optimizer, "get_param_grad", lambda p: p.grad)

    torch.manual_seed(s.seed)
    results = {}
    with full_precision_matmul():
        for step in range(len(s.batches) // micro_batches):
            losses = {}
            step_batches = s.batches[step * micro_batches : (step + 1) * micro_batches]
            if token_weighted:
                tokens = sum(int(s.token_loss.count(batch)) for batch in step_batches)
                if tokens == 0:
                    raise ValueError(
                        f"{s.name}: step {step} has no valid token in any of its {micro_batches} micro-batches; "
                        "the token-normalized loss is undefined"
                    )
            for micro in range(rank, micro_batches, world_size):
                index = step * micro_batches + micro
                batch = s.batches[index]
                raw = module(**_leaves(batch, None))
                if token_weighted:
                    total = s.token_loss.loss(raw, batch).float()
                    booster.backward(total * (world_size / tokens), optimizer)
                    count = int(s.token_loss.count(batch))
                    loss = total / count if count else total
                else:
                    loss = _loss(flatten_tensors(s.select(raw), "out"), grad_seed + index)
                    booster.backward(loss * (world_size / micro_batches), optimizer)
                losses[f"loss/{step}/{micro}"] = loss.detach().float().cpu()
            scale = loss_scale(optimizer)
            if isinstance(optimizer, TorchAMPOptimizer):
                optimizer.unscale_grad()
            if step == 0 and gradients_after_backward:
                # get_param_grad is collective: every rank gathers every parameter
                named = list(inner.named_parameters())
                grads = {n: get_grad(p) for n, p in named if p.requires_grad}
                results.update({f"grad/param/{n}": g.detach().clone() for n, g in grads.items() if g is not None})
                missing = missing_grads(grads, s.unused_params)
            optimizer.step()
            if scale is not None and loss_scale(optimizer) < scale:
                raise RuntimeError(
                    f"step {step}: fp16 gradients overflowed at loss scale {scale}, the step was skipped"
                )
            if step == 0:
                if not gradients_after_backward:
                    grads = full_gradients(plugin, module, optimizer, trainable)
                    results.update({f"grad/param/{n}": g.to(working_dtype) for n, g in grads.items() if g is not None})
                    missing = missing_grads(grads, s.unused_params)
                params = full_parameters(plugin, module, names)
                results.update({f"param/{n}": p.to(working_dtype) for n, p in params.items()})
            optimizer.zero_grad()
            gathered = [None] * world_size
            dist.all_gather_object(gathered, losses)
            for part in gathered:
                results.update(part)
    steps = len(s.batches) // micro_batches
    meta = dict(
        meta,
        plugin=type(plugin).__name__,
        mixed_precision=str(mixed_precision),
        world_size=world_size,
        steps=steps,
        token_weighted=token_weighted,
    )
    return Trace.capture(
        results,
        dtype=str(recipe.dtype).replace("torch.", ""),
        micro_batches=micro_batches,
        missing_grads=missing,
        **contract,
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

    Records ``loss/<step>`` for every step run and, after the last one, ``param/<name>`` and what the checkpoint must
    carry besides them (``_training_state``): the tensors the optimizer updates (float32 master weights where the plugin
    keeps them), its state (Adam's moments and step), the scheduler's state and the fp16 loss scale. Also the random
    number generator states (``rng/...``), which a booster checkpoint does not carry; see ``scenarios.resume``. Each
    rank runs the same batch; data parallelism itself is checked by ``run_data_parallel``.
    """
    module = copy.deepcopy(s.model).to(get_accelerator().get_current_device())
    names = [n for n, _ in module.named_parameters()]
    optimizer = make_optimizer(plugin, optimizer_fn, [p for p in module.parameters() if p.requires_grad])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.25, total_iters=len(s.batches))
    booster = Booster(plugin=plugin, mixed_precision=mixed_precision)
    module, optimizer, _, _, scheduler = booster.boost(module, optimizer, lr_scheduler=scheduler)
    module.train()
    initial_scale = loss_scale(optimizer)
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
            if loss_scale(optimizer) != initial_scale:
                raise RuntimeError(f"step {step}: the fp16 loss scale changed, so the run is not reproducible")
            optimizer.zero_grad()
            scheduler.step()
            results[f"loss/{step}"] = loss.detach().float().cpu()
    results.update({f"param/{n}": p for n, p in full_parameters(plugin, module, names).items()})
    results.update(_training_state(optimizer, scheduler))
    if save is not None:
        booster.save_model(module, os.path.join(save, "model"), shard=True, use_safetensors=True)
        booster.save_optimizer(optimizer, os.path.join(save, "optimizer"), shard=shard_optimizer)
        booster.save_lr_scheduler(scheduler, os.path.join(save, "lr_scheduler.pt"))
    meta = dict(meta, plugin=type(plugin).__name__, mixed_precision=str(mixed_precision), start=start, stop=stop)
    return Trace.capture(results, **_meta(s, grad_seed, meta))


def _training_state(optimizer, scheduler) -> Dict[str, Tensor]:
    # in the plugin's own layout (shards, flat parameters, chunks): resumed and uninterrupted runs of one configuration
    # share it, so the tensors compare position by position without being gathered. Every rank's part, on every rank
    state = {}
    optim = unwrap_optimizer(optimizer)
    for g, group in enumerate(optim.param_groups):
        for i, p in enumerate(group["params"]):
            state[f"optim/{g}.{i}/param"] = p.detach().cpu().clone()
            for key, value in optim.state.get(p, {}).items():
                state[f"optim/{g}.{i}/{key}"] = torch.as_tensor(value).detach().cpu().clone()
    for key, value in scheduler.state_dict().items():
        if isinstance(value, (int, float)) or (isinstance(value, list) and all(isinstance(v, float) for v in value)):
            state[f"scheduler/{key}"] = torch.tensor(value, dtype=torch.float64)
    scale = loss_scale(optimizer)
    if scale is not None:
        state["loss_scale"] = torch.tensor(float(scale), dtype=torch.float64)
    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, state)
        state = {f"rank{r}/{k}": v for r, part in enumerate(gathered) for k, v in part.items()}
    state["rng/cpu"] = torch.get_rng_state()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["rng/cuda"] = torch.cuda.get_rng_state()
    return state


def _unscaled(grad: Optional[Tensor], recipe: Recipe) -> Optional[Tensor]:
    # the loss scale is divided out in the gradient's own dtype, or in float32 for master weights, where the plugin
    # does it while stepping on them: in the low dtype, small gradients would round to subnormals
    if grad is None or not recipe.loss_scale:
        return grad
    if recipe.master_weights:
        return grad.float() * (1.0 / recipe.loss_scale)
    return grad * (1.0 / recipe.loss_scale)


def _trainable(module: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in module.named_parameters() if p.requires_grad]


def _contract(s: Prepared, module: nn.Module, trainable, optimizer, param_map=None) -> Dict[str, Any]:
    # the entry's parameter contract, checked in the run's own model before the plugin wraps it; names are mapped to
    # those of the model under test
    rename = partial(_rename, param_map=param_map)
    trainable = [(rename(n), p) for n, p in trainable if rename(n) is not None]
    return dict(
        not_in_optimizer=not_in_optimizer(trainable, optimizer),
        tie_problems=tie_problems(module, s.tied_params, rename),
    )


def _check_ties_after_boost(contract: Dict[str, Any], s: Prepared, plugin: Plugin, inner: nn.Module) -> None:
    # the plugin must keep tied parameters tied. FSDP's original module holds plain tensors in place of its parameters
    # (its flat parameter sits on the root), so there only the model it was given is checked
    if isinstance(plugin, TorchFSDPPlugin):
        return
    for key, problem in tie_problems(inner, s.tied_params).items():
        contract["tie_problems"].setdefault(key, f"{problem} after boost")


def _loss(outputs: Dict[str, Tensor], grad_seed: int) -> Tensor:
    # upstream_loss divided by the number of output elements: mean-reduced like a real training loss, so that fp16 loss
    # scaling (2**16 by default) does not overflow. Computed in at least float32, as real losses are; otherwise the
    # loss scale itself, flowing back into a float16 loss, is already inf.
    outputs = {k: v.to(torch.promote_types(v.dtype, torch.float32)) for k, v in outputs.items()}
    return upstream_loss(outputs, grad_seed) / sum(out.numel() for out in outputs.values() if out.requires_grad)


def _as_dtype(outputs: Dict[str, Tensor], dtype: torch.dtype, plugin: Plugin) -> Dict[str, Tensor]:
    # GeminiDDP(force_outputs_fp32=True) casts outputs computed in its chunk dtype up to float32; record them in the
    # dtype they were computed in (exact), as the reference does
    if not (isinstance(plugin, GeminiPlugin) and plugin.gemini_config["force_outputs_fp32"]):
        return outputs
    return {k: v.to(dtype) if v.is_floating_point() else v for k, v in outputs.items()}


def _weight_names(module: nn.Module) -> Dict[str, List[str]]:
    # the direct parameters of each module, by attribute name; FSDP replaces them with plain tensors while it runs
    return {
        f"module/{name or '<root>'}": [n for n, _ in sub.named_parameters(recurse=False)]
        for name, sub in module.named_modules()
    }


def _routing_meta(s: Prepared, routing: RoutingRecorder) -> Dict[str, Any]:
    return {"routing": routing.calls} if s.routers is not None else {}


def _meta(s: Prepared, grad_seed: int, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {**dict(model=s.name, seed=s.seed, grad_seed=grad_seed, steps=len(s.batches)), **meta}


def _describe(recipe: Recipe) -> str:
    parts = [str(recipe.dtype).replace("torch.", ""), "autocast" if recipe.autocast else "cast"]
    if recipe.master_weights:
        parts.append("float32 master weights" + (" from the uncast weights" if recipe.master_from_float32 else ""))
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


def _code(*fns) -> str:
    # what decides a run's tensors besides weights and batches, for cache keys: qualified names and, when available,
    # sources (a repr would hold memory addresses)
    parts = []
    for fn in fns:
        if fn is None:
            parts.append("-")
            continue
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            source = ""
        parts.append(f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', type(fn).__qualname__)}\n{source}")
    return "\n".join(parts)


def _build(model_fn: Callable[[], nn.Module], prepare_fn) -> nn.Module:
    model = model_fn()
    if prepare_fn is not None:
        model = prepare_fn(model) or model
    return _disable_dropout(model)


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
    # "<dtype>@<device type>" per tensor: device indices differ between ranks and runs, device types must not
    leaves, _ = tree_flatten(obj)
    names = [f"{str(x.dtype).replace('torch.', '')}@{x.device.type}" for x in leaves if isinstance(x, Tensor)]
    return ",".join(names if ordered else sorted(names)) or "-"


class _RecipeAudit:
    """Records the dtypes and device types a run actually uses, to check that test and reference follow the same
    recipe (e.g. ``float32@cpu`` master weights and Adam moments with CPU offload).

    Per module (first call): input, output and own-weight dtypes, keyed ``module/<name>``. Module names are only
    comparable between identical architectures, so a reference with ``param_map`` records parameters only. Per
    parameter: ``param/<name>`` and ``grad/<name>``. For the optimizer: ``optim/params``, what it updates (the master
    weights where there are any), and ``optim/<state key>``, with the dtypes found. With ``weight_names``
    (``module/<name>`` to attribute names), a module's own weights are looked up by those names instead of its
    registered parameters, for plugins that swap the parameters for other tensors.
    """

    def __init__(self, module: nn.Module, param_map=None, weight_names: Optional[Dict[str, List[str]]] = None):
        self.records: Dict[str, str] = {}
        self.handles = []
        self.weight_names = weight_names
        if param_map is None:
            for name, sub in module.named_modules():
                hook = partial(self._forward, f"module/{name or '<root>'}")
                self.handles.append(sub.register_forward_hook(hook, with_kwargs=True))
        for name, p in module.named_parameters():
            name = _rename(name, param_map)
            if name is not None:
                self.records[f"param/{name}"] = _dtypes(p)
                if p.requires_grad:
                    self.handles.append(p.register_hook(partial(self._grad, f"grad/{name}")))

    def _forward(self, key, module, args, kwargs, output):
        if key not in self.records:
            if self.weight_names is None:
                weights = list(module.parameters(recurse=False))
            else:
                weights = [getattr(module, n, None) for n in self.weight_names.get(key, ())]
                weights = [w for w in weights if isinstance(w, Tensor)]
            # inputs unordered: a rewritten forward (e.g. shardformer's) passes the same tensors in another order
            inputs = _dtypes((args, kwargs), ordered=False)
            self.records[key] = f"in={inputs} out={_dtypes(output)} w={_dtypes(weights)}"

    def _grad(self, key, grad):
        self.records.setdefault(key, _dtypes(grad))

    def record_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        found: Dict[str, set] = {"optim/params": set()}
        for group in optimizer.param_groups:
            found["optim/params"].update(_dtypes(p) for p in group["params"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    found.setdefault(f"optim/{key}", set()).add(_dtypes(value))
        self.records.update({key: ",".join(sorted(v)) for key, v in found.items()})

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
