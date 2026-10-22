"""Token dispatchers for expert parallelism and the data contract they hand to the experts.

A dispatcher moves each token's copies to the ranks that hold its top-k experts (``dispatch``) and brings the expert
outputs back, weighted by the routing weights (``combine``). Experts only see a :class:`DispatchOutput`, so any
dispatcher can be paired with any experts backend.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from ._operation import all_to_all_uneven

__all__ = ["DispatchOutput", "Dispatcher", "NCCLAllToAllDispatcher"]


@dataclass
class DispatchOutput:
    """What a dispatcher gives the experts.

    Attributes:
        tokens: ``[S, H]`` rows for the local experts, grouped by local expert (all rows of local expert 0 first).
            Within an expert, rows keep the order of their tokens in the batch of the whole EP group (source rank
            first, then token position), so expert grads can be compared bitwise with a single-GPU reference.
        tokens_per_expert: ``[E_local]`` int64 row count per local expert, on the same device as ``tokens``.
        tokens_per_expert_cpu: the same counts as a list, when the dispatcher already has them on the host.
        probs: ``[S]`` fp32 routing weight of each row, for dispatchers that let the experts apply the weights. ``None``
            when ``combine`` applies them on the source rank. Experts never apply them otherwise.
        handle: dispatcher-private state needed by ``combine``.
    """

    tokens: Tensor
    tokens_per_expert: Tensor
    tokens_per_expert_cpu: Optional[List[int]] = None
    probs: Optional[Tensor] = None
    handle: Any = None


class Dispatcher(Protocol):
    def dispatch(self, hidden_states: Tensor, topk_idx: Tensor, topk_weights: Tensor) -> DispatchOutput:
        """Send ``hidden_states`` ``[T, H]`` to the ranks of the experts in ``topk_idx`` ``[T, K]`` (global ids)."""

    def combine(self, expert_output: Tensor, dispatched: DispatchOutput) -> Tensor:
        """Return ``[T, H]``: for every token, the sum of its expert outputs weighted by ``topk_weights``."""


class NCCLAllToAllDispatcher:
    """Dispatcher built on NCCL all-to-all, with one host sync per layer (the split sizes NCCL needs on the host).

    The weighted sum is done on the source rank in fp32 with the fp32 routing weights and cast once, which matches
    the grouped_mm path of HF transformers v5 bitwise in the forward pass.

    Args:
        num_experts: total number of experts in the layer.
        ep_group: expert parallel group; experts are split evenly and contiguously over its ranks.
    """

    def __init__(self, num_experts: int, ep_group: Optional[ProcessGroup] = None):
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group) if dist.is_initialized() else 1
        if num_experts % self.ep_size != 0:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by the ep size ({self.ep_size})")
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size

    def dispatch(self, hidden_states: Tensor, topk_idx: Tensor, topk_weights: Tensor) -> DispatchOutput:
        num_tokens, top_k = topk_idx.shape
        flat_idx = topk_idx.reshape(-1)
        # (token, slot) pairs sorted by global expert id, i.e. by destination rank then local expert
        order = torch.sort(flat_idx, stable=True)[1]
        send = hidden_states[order // top_k]
        send_counts = torch.bincount(flat_idx, minlength=self.num_experts).view(self.ep_size, self.num_local_experts)
        if self.ep_size > 1:
            recv_counts = torch.empty_like(send_counts)
            dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)  # [source rank, local expert]
        else:
            recv_counts = send_counts
        # the one host sync: split sizes for NCCL and the per-expert row counts
        sizes = torch.cat([send_counts.sum(1), recv_counts.sum(1), recv_counts.sum(0)]).tolist()
        send_splits = sizes[: self.ep_size]
        recv_splits = sizes[self.ep_size : 2 * self.ep_size]
        tokens_per_expert_cpu = sizes[2 * self.ep_size :]
        if self.ep_size > 1:
            recv, _ = all_to_all_uneven(send, send_splits, recv_splits, self.ep_group)
        else:
            recv = send
        # rows arrive grouped by (source rank, local expert); regroup them by local expert on the GPU
        local_expert = torch.arange(self.num_local_experts, device=recv.device).repeat(self.ep_size)
        row_expert = local_expert.repeat_interleave(recv_counts.reshape(-1), output_size=recv.size(0))
        perm = torch.sort(row_expert, stable=True)[1]
        return DispatchOutput(
            tokens=recv[perm],
            tokens_per_expert=recv_counts.sum(0),
            tokens_per_expert_cpu=tokens_per_expert_cpu,
            handle=(order, perm, send_splits, recv_splits, num_tokens, top_k, topk_weights),
        )

    def combine(self, expert_output: Tensor, dispatched: DispatchOutput) -> Tensor:
        order, perm, send_splits, recv_splits, num_tokens, top_k, topk_weights = dispatched.handle
        out = expert_output[_inverse_permutation(perm)]
        if self.ep_size > 1:
            out, _ = all_to_all_uneven(out, recv_splits, send_splits, self.ep_group)
        weights = topk_weights.reshape(-1)[order].float()
        weighted = (out.float() * weights[:, None])[_inverse_permutation(order)]
        return weighted.view(num_tokens, top_k, out.size(-1)).sum(1).to(expert_output.dtype)


def _inverse_permutation(perm: Tensor) -> Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv
