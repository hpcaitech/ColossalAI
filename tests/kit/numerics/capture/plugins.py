"""What differs between plugins when driving them: the optimizer they accept, where the original module is, and how
to read back full gradients and weights. ``training`` goes through these functions only.

TorchDDP, LowLevelZero and HybridParallel expose the full gradient of each parameter after backward (``param.grad``,
or ``get_param_grad`` for ZeRO). Gemini and FSDP keep gradients in their own sharded, flattened buffers. For those the
gradient is read from the optimizer instead, with Megatron-LM's trick: with Adam's ``beta1 = 0`` the first moment
after a step is exactly the gradient of that step (``exp_avg = 0 * m + 1 * g``), and both plugins can gather the full
optimizer state. Both sides of a comparison then use ``beta1 = 0``, so they still run the same optimizer.
"""

from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch import Tensor, nn

from colossalai.booster.mixed_precision.fp16_torch import TorchAMPOptimizer
from colossalai.booster.plugin import GeminiPlugin, Plugin
from colossalai.booster.plugin.torch_fsdp_plugin import TorchFSDPPlugin

from .recipes import UnsupportedRecipe


def exposes_gradients(plugin: Plugin) -> bool:
    """Whether the full gradient of every parameter can be read after backward."""
    return not isinstance(plugin, (GeminiPlugin, TorchFSDPPlugin))


class FirstMomentIsGradient:
    """Wraps an ``optimizer_fn`` to set Adam's ``beta1`` to 0, so that ``exp_avg`` after a step is that step's gradient.

    A class rather than a closure, so that it can be sent to spawned processes."""

    def __init__(self, optimizer_fn: Callable):
        self.optimizer_fn = optimizer_fn

    def __call__(self, params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
        optimizer = self.optimizer_fn(params)
        for group in optimizer.param_groups:
            if "betas" not in group:
                raise UnsupportedRecipe(
                    f"gradients of Gemini and FSDP are read from Adam's first moment; {type(optimizer).__name__} has none"
                )
            group["betas"] = (0.0, group["betas"][1])
        # TorchFSDPPlugin rebuilds the optimizer from its defaults
        optimizer.defaults["betas"] = (0.0, optimizer.defaults["betas"][1])
        return optimizer


def make_optimizer(plugin: Plugin, optimizer_fn: Callable, params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
    """``optimizer_fn(params)``, translated for plugins that only accept their own optimizers.

    GeminiPlugin requires ColossalAI's ``HybridAdam``; a ``torch.optim.AdamW`` is rebuilt as ``HybridAdam`` with the
    same hyperparameters (``adamw_mode=True``: decoupled weight decay, as AdamW). The reference keeps the torch
    AdamW, an independent implementation of the same update."""
    optimizer = optimizer_fn(params)
    if isinstance(plugin, TorchFSDPPlugin):
        return _for_fsdp(optimizer)
    if not isinstance(plugin, GeminiPlugin):
        return optimizer
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise UnsupportedRecipe(
            "GeminiPlugin accepts ColossalAI's HybridAdam only; the tool translates a torch.optim.AdamW with one "
            f"parameter group, got {type(optimizer).__name__} with {len(optimizer.param_groups)} groups"
        )
    group = optimizer.param_groups[0]
    if group["amsgrad"] or group["maximize"]:
        raise UnsupportedRecipe("AdamW with amsgrad or maximize has no HybridAdam counterpart")
    from colossalai.nn.optimizer import HybridAdam

    return HybridAdam(
        group["params"],
        lr=group["lr"],
        betas=group["betas"],
        eps=group["eps"],
        weight_decay=group["weight_decay"],
        adamw_mode=True,
    )


def _for_fsdp(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    # TorchFSDPPlugin rebuilds the optimizer with optimizer.__init__(params, **optimizer.defaults). In torch 2.13
    # AdamW's defaults hold decoupled_weight_decay, which AdamW.__init__ does not accept; Adam with
    # decoupled_weight_decay=True is the same update and takes it back.
    if type(optimizer) is not torch.optim.AdamW:
        return optimizer
    defaults = dict(optimizer.defaults, decoupled_weight_decay=True)
    return torch.optim.Adam(optimizer.param_groups, **defaults)


def inner_module(plugin: Plugin, boosted: nn.Module) -> nn.Module:
    """The original module inside the plugin's wrappers, with its original module names."""
    inner = boosted.unwrap()
    if isinstance(plugin, TorchFSDPPlugin):
        inner = inner.module  # FullyShardedDataParallel keeps the wrapped module as .module
    return inner


def full_parameters(plugin: Plugin, boosted: nn.Module, names: List[str]) -> Dict[str, Tensor]:
    """The full current value of each named parameter, on every rank. Collective for Gemini and FSDP.

    FSDP returns its float32 sharded parameters and Gemini its chunks; ``training`` casts them to the dtype the plugin
    computes in, which is what the reference records."""
    if isinstance(plugin, TorchFSDPPlugin):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        fsdp = boosted.unwrap()
        with FSDP.state_dict_type(fsdp, StateDictType.FULL_STATE_DICT, FullStateDictConfig(rank0_only=False)):
            state = fsdp.state_dict()
    elif isinstance(plugin, GeminiPlugin):
        state = boosted.state_dict(only_rank_0=False)
    else:
        state = dict(inner_module(plugin, boosted).named_parameters())
    return {name: state[name].detach().clone() for name in names}


def full_gradients(plugin: Plugin, boosted: nn.Module, optimizer, names: List[str]) -> Dict[str, Optional[Tensor]]:
    """For plugins that do not expose gradients: the gradient of the step just taken, from the gathered ``exp_avg``
    (the optimizer must come from ``FirstMomentIsGradient``). ``names`` are the parameters given to the optimizer, in
    order; a parameter without optimizer state got no gradient (``None``). Collective."""
    if isinstance(plugin, TorchFSDPPlugin):
        from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        fsdp = boosted.unwrap()
        with FSDP.state_dict_type(
            fsdp,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
            FullOptimStateDictConfig(rank0_only=False),
        ):
            state = FSDP.optim_state_dict(fsdp, unwrap_optimizer(optimizer))["state"]
        return {name: _exp_avg(state.get(name)) for name in names}
    if isinstance(plugin, GeminiPlugin):
        # GeminiOptimizer numbers the parameters in the order they were given to the optimizer
        return {name: _exp_avg(optimizer.collect_states(i, only_rank_0=False)) for i, name in enumerate(names)}
    raise TypeError(f"{type(plugin).__name__} exposes its gradients; read them after backward")


def loss_scale(optimizer) -> Optional[float]:
    """The current fp16 loss scale of a plugin optimizer, or None without loss scaling."""
    if isinstance(optimizer, TorchAMPOptimizer):
        return optimizer.scaler.get_scale()
    mixin = getattr(optimizer, "mix_precision_mixin", None)
    return getattr(mixin, "loss_scale", None)


def unwrap_optimizer(optimizer) -> torch.optim.Optimizer:
    # plugins and mixed precision may each wrap the torch optimizer
    while hasattr(optimizer, "optim"):
        optimizer = optimizer.optim
    return optimizer


def _exp_avg(state: Optional[dict]) -> Optional[Tensor]:
    if not state or "exp_avg" not in state:
        return None
    return state["exp_avg"].detach().clone()
