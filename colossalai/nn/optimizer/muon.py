"""Muon: momentum orthogonalized by Newton-Schulz, with a built-in AdamW for non-matrix parameters.

The Muon update follows ``torch.optim.Muon`` (torch>=2.9) so the two are numerically interchangeable; it is
reimplemented here because ColossalAI still supports older torch. Non-matrix parameters (embeddings, output head,
norms, biases) go to the AdamW branch of the same optimizer, as in KellerJordan/Muon's ``MuonWithAuxAdam`` and
Moonlight, so a Booster still receives a single optimizer.

References:
    - https://kellerjordan.github.io/posts/muon/
    - Muon is Scalable for LLM Training, https://arxiv.org/abs/2502.16982
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["Muon", "muon_param_groups", "zeropower_via_newtonschulz"]

# Quintic Newton-Schulz coefficients from KellerJordan/Muon, also torch.optim.Muon's defaults.
DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
ADJUST_LR_FNS = ("original", "match_rms_adamw")


def zeropower_via_newtonschulz(
    grad: Tensor,
    ns_coefficients: Tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
    ns_steps: int = 5,
    eps: float = 1e-7,
) -> Tensor:
    """Approximately orthogonalize a 2D matrix with a quintic Newton-Schulz iteration.

    With the default coefficients the result is U S' V^T with S' roughly in [0.5, 1.5] rather than the exact polar
    factor U V^T, which works as well in training. The iteration runs in bf16, as in every reference implementation.
    """
    if grad.ndim != 2:
        raise ValueError(f"Newton-Schulz expects a 2D matrix, got shape {tuple(grad.shape)}")
    a, b, c = ns_coefficients
    x = grad.bfloat16()
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        x = x.T
    # Scale the spectral norm to at most 1; out of place so a low-precision input buffer is never modified.
    x = x / x.norm().clamp(min=eps)
    for _ in range(ns_steps):
        gram = x @ x.T
        poly = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.addmm(x, poly, x, beta=a)
    if transposed:
        x = x.T
    return x


def _adjust_lr(lr: float, adjust_lr_fn: Optional[str], shape: torch.Size) -> float:
    rows, cols = shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        # KellerJordan/Muon: keep the update RMS consistent across rectangular matrices.
        return lr * math.sqrt(max(1.0, rows / cols))
    # Moonlight: match the update RMS of AdamW so AdamW's lr and weight decay carry over.
    return lr * 0.2 * math.sqrt(max(rows, cols))


class Muon(Optimizer):
    """Muon for 2D hidden-layer weights plus AdamW for every other parameter, in one optimizer.

    Each parameter group sets ``use_muon``; ``muon_param_groups(model)`` builds both groups from a model. Groups
    without the flag are Muon groups. Muon groups take ``lr``, ``weight_decay``, ``momentum``, ``nesterov``,
    ``ns_coefficients``, ``ns_steps``, ``eps`` and ``adjust_lr_fn``; AdamW groups take ``lr``, ``weight_decay``,
    ``betas`` and ``eps``. Missing keys fall back to the constructor arguments, with ``adamw_eps`` as the AdamW groups'
    ``eps``.

    Args:
        params: parameters or parameter groups.
        lr (float): learning rate shared by both branches unless a group overrides it (default: 1e-3).
        weight_decay (float): decoupled weight decay, applied as ``p *= 1 - lr * weight_decay`` (default: 0.1).
        momentum (float): Muon momentum, kept as an exponential moving average like torch.optim.Muon (default: 0.95).
        nesterov (bool): use Nesterov momentum for Muon (default: True).
        ns_coefficients (tuple): Newton-Schulz coefficients (a, b, c) (default: KellerJordan's).
        ns_steps (int): Newton-Schulz iterations (default: 5).
        eps (float): lower bound of the norm used to scale the Newton-Schulz input (default: 1e-7).
        adjust_lr_fn (str, optional): ``"original"`` scales the Muon lr by sqrt(max(1, rows / cols));
            ``"match_rms_adamw"`` by 0.2 * sqrt(max(rows, cols)) so AdamW hyperparameters carry over.
            None means ``"original"`` (default: None).
        betas (tuple): AdamW betas (default: (0.9, 0.95)).
        adamw_eps (float): AdamW epsilon (default: 1e-8).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: Tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
        ns_steps: int = 5,
        eps: float = 1e-7,
        adjust_lr_fn: Optional[str] = None,
        betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0 < ns_steps < 100:
            raise ValueError(f"ns_steps must be in [1, 99], got {ns_steps}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ADJUST_LR_FNS:
            raise ValueError(f"adjust_lr_fn must be one of {ADJUST_LR_FNS} or None, got {adjust_lr_fn!r}")
        self._muon_defaults = dict(
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=tuple(ns_coefficients),
            ns_steps=ns_steps,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
        )
        self._adamw_defaults = dict(betas=tuple(betas), eps=adamw_eps)
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

    def add_param_group(self, param_group: Dict) -> None:
        use_muon = param_group.setdefault("use_muon", True)
        for key, value in (self._muon_defaults if use_muon else self._adamw_defaults).items():
            param_group.setdefault(key, value)
        params = param_group["params"]
        if isinstance(params, Tensor):
            params = [params]
        elif not isinstance(params, set):  # leave sets for torch to reject: their order is not deterministic
            params = list(params)
        param_group["params"] = params
        if use_muon:
            for p in params:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters, got shape {tuple(p.shape)}; "
                        "put it in a group with use_muon=False (see muon_param_groups)"
                    )
        super().add_param_group(param_group)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: Dict) -> None:
        lr, weight_decay, momentum = group["lr"], group["weight_decay"], group["momentum"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.ndim != 2:
                # A sharded 1D view, e.g. from LowLevelZeRO's master parameter slices.
                raise RuntimeError(
                    f"Muon needs the full 2D gradient but got shape {tuple(grad.shape)}; "
                    "ZeRO / Gemini / FSDP sharding is not supported yet"
                )
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
            buf = state["momentum_buffer"]
            buf.lerp_(grad, 1 - momentum)
            update = grad.lerp(buf, momentum) if group["nesterov"] else buf
            update = zeropower_via_newtonschulz(update, group["ns_coefficients"], group["ns_steps"], group["eps"])
            p.mul_(1 - lr * weight_decay)
            p.add_(update, alpha=-_adjust_lr(lr, group["adjust_lr_fn"], p.shape))

    def _adamw_step(self, group: Dict) -> None:
        lr, weight_decay, eps = group["lr"], group["weight_decay"], group["eps"]
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            # Same arithmetic as torch.optim.AdamW's single-tensor path.
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            p.mul_(1 - lr * weight_decay)
            p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)


def muon_param_groups(model: nn.Module) -> List[Dict]:
    """Split a model's trainable parameters into a Muon group and an AdamW group.

    Muon takes 2D weights of hidden layers. Embeddings, the output head returned by ``get_output_embeddings()``
    (tied or not) and every non-2D parameter go to AdamW. Setting ``param.use_muon`` to True or False overrides
    the rule for that parameter. Empty groups are dropped, so pass the result straight to ``Muon``.
    """
    excluded = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            excluded.add(id(module.weight))
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    head = get_output_embeddings() if callable(get_output_embeddings) else None
    if head is not None and getattr(head, "weight", None) is not None:
        excluded.add(id(head.weight))

    muon_params, adamw_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        use_muon = getattr(p, "use_muon", None)
        if use_muon is None:
            use_muon = p.ndim == 2 and id(p) not in excluded
        (muon_params if use_muon else adamw_params).append(p)

    groups = []
    if muon_params:
        groups.append({"params": muon_params, "use_muon": True})
    if adamw_params:
        groups.append({"params": adamw_params, "use_muon": False})
    return groups
