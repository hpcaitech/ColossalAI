"""EPMoELayer: the expert-parallel part of an MoE block, shared by all models.

An MoE block is split into Router -> dispatch -> experts -> combine. Models keep their own router (top-k selection,
grouped routing, ...) and hand its result to this layer, which runs the other three steps with a pluggable
:class:`~colossalai.moe.dispatcher.Dispatcher` and :class:`~colossalai.moe.experts.Experts` backend.
"""

import math
from dataclasses import replace
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from colossalai.tensor.moe_tensor.api import ExpertGradScale, set_expert_grad_scale, set_moe_tensor_ep_group

from ._operation import EPGradScalerIn, EPGradScalerOut
from .dispatcher import Dispatcher, NCCLAllToAllDispatcher
from .experts import ActFn, Experts, GroupedMMExperts

__all__ = ["EPMoELayer"]


class EPMoELayer(nn.Module):
    """Routed experts of an MoE block with expert parallelism.

    Each rank of ``ep_group`` holds ``num_experts / ep_size`` consecutive experts as two parameters in the
    transformers v5 layout: ``gate_up_proj`` ``[E_local, 2 * intermediate_size, hidden_size]`` and ``down_proj``
    ``[E_local, hidden_size, intermediate_size]``.

    Gradient contract: the gradient of an expert weight on one rank sums the contributions of all tokens the EP group
    sent to it. Where its ``1 / ep_size`` factor is applied is recorded on both parameters
    (:class:`~colossalai.tensor.moe_tensor.api.ExpertGradScale`):

    - ``IN_FORWARD`` (default): this layer applies it in backward, like the existing EP models, so the current
      plugins average expert grads correctly.
    - ``IN_REDUCTION``: this layer does not; the gradient reduction must divide expert grads by
      ``replica_group_size * ep_size``. Only use it with a plugin that reads this tag, or expert grads are ``ep_size``
      times too large.

    Args:
        num_experts: total number of routed experts.
        hidden_size: model hidden size.
        intermediate_size: hidden size of each expert MLP.
        ep_group: expert parallel group; ``None`` means the default group.
        act_fn: activation on the gate projection (SwiGLU: ``act_fn(gate) * up``).
        dispatcher: defaults to :class:`~colossalai.moe.dispatcher.NCCLAllToAllDispatcher`.
        experts: defaults to :class:`~colossalai.moe.experts.GroupedMMExperts`.
        expert_grad_scale: see the gradient contract above.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        ep_group: Optional[ProcessGroup] = None,
        act_fn: ActFn = F.silu,
        dispatcher: Optional[Dispatcher] = None,
        experts: Optional[Experts] = None,
        expert_grad_scale: ExpertGradScale = ExpertGradScale.IN_FORWARD,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if ep_group is None and dist.is_initialized():
            ep_group = dist.group.WORLD
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group) if dist.is_initialized() else 1
        self.ep_rank = dist.get_rank(ep_group) if dist.is_initialized() else 0
        if num_experts % self.ep_size != 0:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by the ep size ({self.ep_size})")
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = act_fn
        self.expert_grad_scale = ExpertGradScale(expert_grad_scale)
        self.dispatcher = dispatcher if dispatcher is not None else NCCLAllToAllDispatcher(num_experts, ep_group)
        self.experts = experts if experts is not None else GroupedMMExperts()

        factory = dict(device=device, dtype=dtype)
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * intermediate_size, hidden_size, **factory)
        )
        self.down_proj = nn.Parameter(torch.empty(self.num_local_experts, hidden_size, intermediate_size, **factory))
        self.reset_parameters()
        for param in (self.gate_up_proj, self.down_proj):
            set_moe_tensor_ep_group(param, ep_group)
            set_expert_grad_scale(param, self.expert_grad_scale)

    @property
    def local_expert_range(self) -> range:
        """Global ids of the experts held by this rank."""
        start = self.ep_rank * self.num_local_experts
        return range(start, start + self.num_local_experts)

    def reset_parameters(self) -> None:
        # each expert initialised like nn.Linear
        for weight in (self.gate_up_proj, self.down_proj):
            bound = 1 / math.sqrt(weight.size(-1))
            nn.init.uniform_(weight, -bound, bound)

    @torch.no_grad()
    def load_full_weights(self, gate_up_proj: Tensor, down_proj: Tensor) -> None:
        """Copy this rank's experts out of the weights of all experts (``[E, 2I, H]`` and ``[E, H, I]``).

        Only ``.data`` is written, so the parameter objects that optimizers and ZeRO already hold stay valid.
        """
        local = slice(self.local_expert_range.start, self.local_expert_range.stop)
        self.gate_up_proj.data.copy_(gate_up_proj[local])
        self.down_proj.data.copy_(down_proj[local])

    def forward(self, hidden_states: Tensor, topk_idx: Tensor, topk_weights: Tensor) -> Tensor:
        """Run the routed experts.

        Args:
            hidden_states: ``[..., hidden_size]`` input of the MoE block.
            topk_idx: ``[..., top_k]`` global expert ids chosen by the router for each token.
            topk_weights: ``[..., top_k]`` routing weights, ideally fp32; gradients flow back through them.

        Returns:
            Tensor of the shape and dtype of ``hidden_states``: per token, the weighted sum of its experts' outputs.
        """
        shape = hidden_states.shape
        top_k = topk_idx.size(-1)
        dispatched = self.dispatcher.dispatch(
            hidden_states.reshape(-1, self.hidden_size), topk_idx.reshape(-1, top_k), topk_weights.reshape(-1, top_k)
        )
        scale_in_forward = self.expert_grad_scale == ExpertGradScale.IN_FORWARD and self.ep_size > 1
        if scale_in_forward:
            # backward: divide the expert weight grads by ep_size, leave the input grads unchanged
            dispatched = replace(dispatched, tokens=EPGradScalerIn.apply(dispatched.tokens, self.ep_size))
        output = self.experts(dispatched, self.gate_up_proj, self.down_proj, self.act_fn)
        if scale_in_forward:
            output = EPGradScalerOut.apply(output, self.ep_size)
        return self.dispatcher.combine(output, dispatched).view(shape)

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, num_local_experts={self.num_local_experts}, "
            f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, ep_size={self.ep_size}, "
            f"expert_grad_scale={self.expert_grad_scale.value}"
        )
