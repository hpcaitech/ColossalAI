"""Muon: momentum orthogonalized by Newton-Schulz, with a built-in AdamW for non-matrix parameters.

The Muon update follows ``torch.optim.Muon`` (torch>=2.9) so the two are numerically interchangeable; it is
reimplemented here because ColossalAI still supports older torch. Non-matrix parameters (embeddings, output head,
norms, biases) go to the AdamW branch of the same optimizer, as in KellerJordan/Muon's ``MuonWithAuxAdam`` and
Moonlight, so a Booster still receives a single optimizer.

Under LowLevelZeRO (stage 1 and 2) each rank holds a flat 1/N slice of every parameter, while Newton-Schulz needs whole
matrices. As in Moonlight's distributed Muon, each rank updates the momentum of its own slices, the bf16
Newton-Schulz inputs are all-gathered over the data-parallel group, and every rank orthogonalizes the whole matrices
and keeps the slice of each update it owns. The gathered input is the bf16 cast the unsharded iteration makes anyway,
so a sharded step computes the unsharded one, at the cost of one bf16 all-gather of the Muon parameters per step and
every rank orthogonalizing all of them.

References:
    - https://kellerjordan.github.io/posts/muon/
    - Muon is Scalable for LLM Training, https://arxiv.org/abs/2502.16982
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from colossalai.accelerator import get_accelerator
from colossalai.interface.optimizer import DistributedOptim

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


class Muon(DistributedOptim):
    """Muon for 2D hidden-layer weights plus AdamW for every other parameter, in one optimizer.

    Each parameter group sets ``use_muon``; ``muon_param_groups(model)`` builds both groups from a model. Groups
    without the flag are Muon groups. Muon groups take ``lr``, ``weight_decay``, ``momentum``, ``nesterov``,
    ``ns_coefficients``, ``ns_steps``, ``eps`` and ``adjust_lr_fn``; AdamW groups take ``lr``, ``weight_decay``,
    ``betas`` and ``eps``. Missing keys fall back to the constructor arguments, with ``adamw_eps`` as the AdamW groups'
    ``eps``.

    ``LowLevelZeroPlugin`` and ``HybridParallelPlugin`` call ``setup_distributed`` on this very object (it is not
    replaced, so an LR scheduler built on it before ``boost`` keeps driving it), after which ZeRO's parameter shards are
    orthogonalized as whole matrices (see the module docstring). The AdamW branch is elementwise and runs on the
    shards as they are. Tensor parallelism, Gemini and FSDP are not supported.

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

    # Under ZeRO at most this many elements of whole matrices are gathered at once (512 MiB in bf16), which bounds the
    # extra memory of a step.
    bucket_numel = 1 << 28

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
        self.dp_group: Optional[dist.ProcessGroup] = None
        self.shard_to_working_param: Dict[int, Tensor] = {}

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

    def setup_distributed(
        self,
        tp_group: Optional[dist.ProcessGroup] = None,
        dp_group: Optional[dist.ProcessGroup] = None,
        shard_to_working_param: Optional[Dict] = None,
        padding_map: Optional[Dict] = None,
        is_zero: Optional[bool] = False,
    ):
        """Record the data-parallel group and, for ZeRO, which whole parameter each shard belongs to.

        Arguments:
            tp_group (dist.ProcessGroup): tensor-parallel group; only a group of size 1 is supported.
            dp_group (dist.ProcessGroup): the group ZeRO shards over (default: the whole world).
            shard_to_working_param (Dict): id of each shard to the parameter used in forward and backward.
            padding_map (Dict): unused; the padding follows from the parameter's size and the group size.
            is_zero (bool): unused; shards are recognized by their shape.
        """
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            raise NotImplementedError("Muon does not support tensor parallelism yet; use tp_size=1")
        self.dp_group = dp_group
        self.shard_to_working_param = dict(shard_to_working_param or {})
        world_size = dist.get_world_size(dp_group)
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for p in group["params"]:
                working = self._working_param(p)
                if working.shape == p.shape:
                    continue
                if working.ndim != 2 or p.numel() != -(-working.numel() // world_size):
                    raise ValueError(
                        f"a {tuple(working.shape)} Muon parameter arrives as a shard of {p.numel()} elements, not one "
                        f"of {world_size} equal slices: its shards do not span the data-parallel group (e.g. expert "
                        "parameters sharded over their own group), which Muon does not support"
                    )

    def _working_param(self, p: Tensor) -> Tensor:
        return self.shard_to_working_param.get(id(p), p)

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
        shards = []
        for p in group["params"]:
            if p.grad is None:
                continue
            if self._working_param(p).shape == p.shape:
                self._step_matrix(p, group)
            else:
                shards.append(p)
        # every rank holds the same shards in the same order, so the gathers line up
        for bucket in self._buckets(shards):
            inputs = [self._momentum_update(p, group) for p in bucket]
            for p, update in zip(bucket, self._orthogonalize_shards(bucket, inputs, group)):
                self._apply_update(p, update, self._working_param(p).shape, group)

    def _step_matrix(self, p: Tensor, group: Dict) -> None:
        if p.grad.ndim != 2:
            # A sharded 1D view that setup_distributed was not told about.
            raise RuntimeError(
                f"Muon needs the full 2D gradient but got shape {tuple(p.grad.shape)}; under ZeRO the plugin has to "
                "call setup_distributed (LowLevelZeroPlugin does), and Gemini / FSDP are not supported"
            )
        update = zeropower_via_newtonschulz(
            self._momentum_update(p, group), group["ns_coefficients"], group["ns_steps"], group["eps"]
        )
        self._apply_update(p, update, p.shape, group)

    def _momentum_update(self, p: Tensor, group: Dict) -> Tensor:
        """Update ``p``'s momentum buffer and return the Newton-Schulz input; elementwise, so shards work too."""
        grad, momentum = p.grad, group["momentum"]
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1 - momentum)
        return grad.lerp(buf, momentum) if group["nesterov"] else buf

    @staticmethod
    def _apply_update(p: Tensor, update: Tensor, shape: torch.Size, group: Dict) -> None:
        """Apply weight decay and the orthogonalized ``update``; ``shape`` is the whole matrix's, for the lr scale."""
        lr = group["lr"]
        p.mul_(1 - lr * group["weight_decay"])
        p.add_(update, alpha=-_adjust_lr(lr, group["adjust_lr_fn"], shape))

    def _buckets(self, shards: List[Tensor]) -> List[List[Tensor]]:
        buckets, numel = [], 0
        for p in shards:
            size = self._working_param(p).numel()
            if not buckets or numel + size > self.bucket_numel:
                buckets.append([])
                numel = 0
            buckets[-1].append(p)
            numel += size
        return buckets

    def _orthogonalize_shards(self, shards: List[Tensor], inputs: List[Tensor], group: Dict) -> List[Tensor]:
        """Gather the whole Newton-Schulz inputs of ``shards``, orthogonalize them and return each rank's slices."""
        world_size, rank = dist.get_world_size(self.dp_group), dist.get_rank(self.dp_group)
        # gloo gathers CPU tensors; NCCL (and Newton-Schulz) want the accelerator
        if dist.get_backend(self.dp_group) == dist.Backend.GLOO:
            device = torch.device("cpu")
        else:
            device = get_accelerator().get_current_device()
        local = torch.cat([x.to(device=device, dtype=torch.bfloat16).flatten() for x in inputs])
        gathered = torch.empty(world_size * local.numel(), dtype=local.dtype, device=device)
        dist.all_gather_into_tensor(gathered, local, group=self.dp_group)
        # row r holds rank r's slices of every matrix, one after another
        gathered = gathered.view(world_size, local.numel())
        updates, offset = [], 0
        for p in shards:
            working, n = self._working_param(p), p.numel()
            matrix = gathered[:, offset : offset + n].reshape(-1)[: working.numel()].view(working.shape)
            offset += n
            update = zeropower_via_newtonschulz(matrix, group["ns_coefficients"], group["ns_steps"], group["eps"])
            update = F.pad(update.flatten(), (0, world_size * n - working.numel())).view(world_size, n)[rank]
            updates.append(update.to(p.device))
        return updates

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
