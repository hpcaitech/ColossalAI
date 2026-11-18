"""OrScale: Muon's orthogonalized update scaled by a layer-wise trust ratio, with a built-in AdamW.

For a matrix W with Nesterov momentum M, the orthogonalized direction Q = PolarExpress(M) and shape scale s:

    r = clip(||W||_F / (c * ||wd * W + s * Q||_F + eps), r_min, r_max)
    W <- W - lr * r * (wd * W + s * Q)

so the weight decay is coupled into the trust-ratio-scaled step. The two variants of the paper:

    OrScale     s = 1, c = 1
    OrScale-LM  s = 0.2 * sqrt(max(rows, cols)) (Moonlight's), and c set per layer at its first step so that r = 1 there

The arithmetic follows the official implementation (https://github.com/NUS-HPC-AI-Lab/OrScale, commit a45fc2b)
statement by statement: accumulated float32 momentum (not ``Muon``'s moving average), bf16 Polar Express
orthogonalization, float32 norms. Run eagerly, the two give the same bits. The official code compiles its
orthogonalization with ``torch.compile`` on CUDA, which this one does not; that alone moves the result by 1-3%, as much
as bf16 moves it from a float64 iteration, and so does any other change of the bf16 rounding, such as Muon's moving
average in place of the accumulated momentum. Its two optimizers (OrScale for matrices, AdamW for the rest) are one
here, as in ``Muon``, so a Booster still receives a single optimizer; its ablation variants and its flattening of
convolution weights are left out.

Under LowLevelZeRO the matrices are orthogonalized whole, as ``Muon`` does. The trust ratio needs two norms of the
whole matrix: each rank sums the squares over its slice and one all-reduce of two numbers per matrix adds them up.
Sums in a different order round differently, so a sharded step equals the unsharded one up to float32 rounding of
the trust ratio (about 1e-7), not bit for bit.

Reference:
    - OrScale: Orthogonalised Optimization with Layer-Wise Trust-Ratio Scaling, https://arxiv.org/abs/2605.07815
    - The Polar Express, https://arxiv.org/abs/2505.16932
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from .muon import Muon

__all__ = ["OrScale", "polar_express"]

# One (a, b, c) per iteration, from the official OrScale implementation (orscale/optim/newton_schulz.py).
POLAR_EXPRESS_COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)
VARIANTS = ("orscale", "orscale_lm")
MOONLIGHT_RMS_CONSTANT = 0.2


def polar_express(grad: Tensor, ns_steps: int = 5) -> Tensor:
    """Approximate the polar factor U V^T of a 2D matrix with the Polar Express iteration, in bf16.

    Unlike ``zeropower_via_newtonschulz`` each iteration has its own coefficients, which converge to U V^T itself, and
    the input is scaled by 1 / (1.02 * ||grad||_F + 1e-6).
    """
    if grad.ndim != 2:
        raise ValueError(f"Polar Express expects a 2D matrix, got shape {tuple(grad.shape)}")
    if not 0 < ns_steps <= len(POLAR_EXPRESS_COEFFICIENTS):
        raise ValueError(f"ns_steps must be in [1, {len(POLAR_EXPRESS_COEFFICIENTS)}], got {ns_steps}")
    x = grad.bfloat16()
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        x = x.T
    x = (x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)).contiguous()
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS[:ns_steps]:
        gram = x @ x.T
        poly = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.addmm(x, poly, x, beta=a)
    if transposed:
        x = x.T
    return x


class OrScale(Muon):
    """OrScale or OrScale-LM for 2D hidden-layer weights plus AdamW for every other parameter, in one optimizer.

    Parameter groups work as in ``Muon``: each sets ``use_muon`` (``muon_param_groups(model)`` builds both groups);
    groups without the flag are OrScale groups. OrScale groups take ``lr``, ``weight_decay``, ``momentum``,
    ``variant``, ``r_min``, ``r_max``, ``eps``, ``ns_steps`` and ``c_denom``; AdamW groups take ``lr``,
    ``weight_decay``, ``betas`` and ``eps``. Missing keys fall back to the constructor arguments, with ``adamw_eps`` as
    the AdamW groups' ``eps``. The defaults are the official ones for the matrices, where ``lr`` is far larger than
    AdamW's, so give the AdamW group its own ``lr`` (the official default is a tenth of it).

    Like ``Muon``, it runs under ``TorchDDPPlugin`` and ``LowLevelZeroPlugin`` (ZeRO stage 1 and 2, see the module
    docstring); tensor parallelism, Gemini and FSDP are not supported.

    Args:
        params: parameters or parameter groups.
        lr (float): learning rate shared by both branches unless a group overrides it (default: 0.02).
        weight_decay (float): weight decay; for the matrices it is coupled into the trust-ratio-scaled step, for the
            AdamW groups it is decoupled (default: 0.0).
        momentum (float): Nesterov momentum of the matrices, accumulated as ``M = momentum * M + grad`` in float32
            (default: 0.95).
        variant (str): ``"orscale"`` or ``"orscale_lm"`` (default: ``"orscale"``).
        r_min (float): lower clipping bound of the trust ratio (default: 0.5).
        r_max (float): upper clipping bound of the trust ratio (default: 1.5). The official recommendation is
            [0.5, 1.5] for ``"orscale"`` and [0.1, 5.0] for ``"orscale_lm"``.
        eps (float): added to the denominator of the trust ratio (default: 1e-6).
        ns_steps (int): Polar Express iterations, at most 5 (default: 5).
        c_denom (float, optional): ``"orscale_lm"`` only. The constant c of every layer; None calibrates it per layer
            at the layer's first step as ``||W|| / ||wd * W + s * Q||`` and keeps it in the optimizer state
            (default: None).
        betas (tuple): AdamW betas (default: (0.9, 0.95)).
        adamw_eps (float): AdamW epsilon (default: 1e-8).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        variant: str = "orscale",
        r_min: float = 0.5,
        r_max: float = 1.5,
        eps: float = 1e-6,
        ns_steps: int = 5,
        c_denom: Optional[float] = None,
        betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        orscale_defaults = dict(
            momentum=momentum, variant=variant, r_min=r_min, r_max=r_max, eps=eps, ns_steps=ns_steps, c_denom=c_denom
        )
        self._init_groups(params, lr, weight_decay, orscale_defaults, dict(betas=tuple(betas), eps=adamw_eps))

    def add_param_group(self, param_group: Dict) -> None:
        if param_group.setdefault("use_muon", True):
            group = {**self._muon_defaults, **param_group}
            if not 0.0 <= group["momentum"] < 1.0:
                raise ValueError(f"Invalid momentum: {group['momentum']}")
            if group["variant"] not in VARIANTS:
                raise ValueError(f"variant must be one of {VARIANTS}, got {group['variant']!r}")
            if not 0.0 <= group["r_min"] <= group["r_max"]:
                raise ValueError(f"need 0 <= r_min <= r_max, got r_min={group['r_min']}, r_max={group['r_max']}")
            if not 0 < group["ns_steps"] <= len(POLAR_EXPRESS_COEFFICIENTS):
                raise ValueError(f"ns_steps must be in [1, {len(POLAR_EXPRESS_COEFFICIENTS)}], got {group['ns_steps']}")
            if group["c_denom"] is not None and group["c_denom"] <= 0.0:
                raise ValueError(f"c_denom must be positive or None, got {group['c_denom']}")
        super().add_param_group(param_group)

    def _momentum_update(self, p: Tensor, group: Dict) -> Tensor:
        """Update ``p``'s momentum buffer and return the Polar Express input; elementwise, so shards work too."""
        # accumulated in float32 whatever the parameter's dtype
        grad, momentum = p.grad.float(), group["momentum"]
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
        buf = state["momentum_buffer"]
        buf.mul_(momentum).add_(grad)
        return buf.mul(momentum).add_(grad)

    def _orthogonalize(self, matrix: Tensor, group: Dict) -> Tensor:
        return polar_express(matrix, group["ns_steps"])

    @staticmethod
    def _shape_scale(shape: torch.Size, group: Dict) -> float:
        if group["variant"] == "orscale_lm":
            return MOONLIGHT_RMS_CONSTANT * math.sqrt(max(shape))
        return 1.0

    def _apply_update(self, p: Tensor, update: Tensor, group: Dict) -> None:
        scale, weight_decay = self._shape_scale(p.shape, group), group["weight_decay"]
        weight_norm = p.norm().float()
        step_norm = (weight_decay * p.float() + scale * update.float()).norm()
        self._scaled_step(p, update, self._trust_ratio(p, weight_norm, step_norm, p.shape, group), scale, group)

    def _apply_shard_updates(self, shards: List[Tensor], updates: List[Tensor], group: Dict) -> None:
        weight_decay = group["weight_decay"]
        scales = [self._shape_scale(self._working_param(p).shape, group) for p in shards]
        # the padding of a matrix's last slice is zero in the weights and in the update, and stays zero
        squares = []
        for p, update, scale in zip(shards, updates, scales):
            weight = p.float()
            squares.append(weight.pow(2).sum())
            squares.append((weight_decay * weight + scale * update.float()).pow(2).sum())
        squares = torch.stack(squares).to(self._collective_device())
        dist.all_reduce(squares, group=self.dp_group)
        norms = squares.sqrt().cpu().view(-1, 2)
        for p, update, scale, (weight_norm, step_norm) in zip(shards, updates, scales, norms):
            ratio = self._trust_ratio(p, weight_norm, step_norm, self._working_param(p).shape, group)
            self._scaled_step(p, update, ratio, scale, group)

    def _trust_ratio(self, p: Tensor, weight_norm: Tensor, step_norm: Tensor, shape: torch.Size, group: Dict) -> float:
        """The clipped trust ratio from the float32 norms of the whole matrix and of ``wd * W + s * Q``."""
        eps = group["eps"]
        if group["variant"] == "orscale_lm":
            state = self.state[p]
            if "c_denom" not in state:
                c_denom = group["c_denom"]
                if c_denom is None:
                    weight_init, step_init = weight_norm.item(), step_norm.item()
                    if weight_init < eps or step_init < eps:
                        # a layer initialized to (almost) zero: fall back to Moonlight's scale instead of a runaway c
                        c_denom = MOONLIGHT_RMS_CONSTANT * math.sqrt(max(shape))
                    else:
                        c_denom = weight_init / step_init
                # a float, not a 0-dim tensor: LowLevelZeRO's checkpoints treat every tensor in the state as sharded
                state["c_denom"] = float(c_denom)
            step_norm = state["c_denom"] * step_norm
        ratio = (weight_norm / (step_norm + eps)).item()
        return max(group["r_min"], min(group["r_max"], ratio))

    @staticmethod
    def _scaled_step(p: Tensor, update: Tensor, ratio: float, scale: float, group: Dict) -> None:
        # elementwise, so the same for a matrix and a shard
        lr, weight_decay = group["lr"], group["weight_decay"]
        if weight_decay != 0.0:
            p.add_(p, alpha=-lr * ratio * weight_decay)
        p.add_(update.to(p.dtype), alpha=-lr * ratio * scale)
