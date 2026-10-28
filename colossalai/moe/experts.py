"""Experts backends: the SwiGLU MLPs of the local experts, run on the rows of a :class:`DispatchOutput`.

Backends are stateless; the weights belong to the layer and use the transformers v5 layout:
``gate_up_proj`` ``[E_local, 2 * I, H]`` (gate rows first) and ``down_proj`` ``[E_local, H, I]``.
Outputs are not multiplied by the routing weights: the dispatcher's ``combine`` does that.
"""

from typing import Callable, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from .dispatcher import DispatchOutput

__all__ = ["Experts", "GroupedMMExperts", "ReferenceExperts"]

ActFn = Callable[[Tensor], Tensor]


class Experts(Protocol):
    def __call__(self, dispatched: DispatchOutput, gate_up_proj: Tensor, down_proj: Tensor, act_fn: ActFn) -> Tensor:
        """Return ``[S, H]``, one output row per row of ``dispatched.tokens``, in the same order."""


def _torch_version() -> tuple:
    return tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])


class GroupedMMExperts:
    """All local experts in two ``torch._grouped_mm`` calls (torch >= 2.8).

    Before torch 2.11, ``_grouped_mm`` needs every group's row count to be a multiple of 8 (16 bytes in bf16), so the
    rows are padded per expert and the padding is dropped afterwards.
    """

    def __init__(self):
        if _torch_version() < (2, 8) or not hasattr(torch, "_grouped_mm"):
            raise RuntimeError(f"GroupedMMExperts needs torch >= 2.8, got {torch.__version__}; use ReferenceExperts")
        self.align = 1 if _torch_version() >= (2, 11) else 8

    def __call__(self, dispatched: DispatchOutput, gate_up_proj: Tensor, down_proj: Tensor, act_fn: ActFn) -> Tensor:
        tokens, counts = dispatched.tokens, dispatched.tokens_per_expert
        if self.align == 1:
            return self._mlp(tokens, counts, gate_up_proj, down_proj, act_fn)
        num_experts = counts.numel()
        padded_counts = (counts + self.align - 1) // self.align * self.align
        start = torch.cumsum(counts, 0) - counts
        padded_start = torch.cumsum(padded_counts, 0) - padded_counts
        row_expert = torch.arange(num_experts, device=tokens.device).repeat_interleave(
            counts, output_size=tokens.size(0)
        )
        dest = torch.arange(tokens.size(0), device=tokens.device) - start[row_expert] + padded_start[row_expert]
        padded = tokens.new_zeros(tokens.size(0) + (self.align - 1) * num_experts, tokens.size(1))
        padded = padded.index_copy(0, dest, tokens)
        return self._mlp(padded, padded_counts, gate_up_proj, down_proj, act_fn)[dest]

    @staticmethod
    def _mlp(tokens: Tensor, counts: Tensor, gate_up_proj: Tensor, down_proj: Tensor, act_fn: ActFn) -> Tensor:
        offs = torch.cumsum(counts, 0).to(torch.int32)
        gate, up = torch._grouped_mm(tokens, gate_up_proj.transpose(1, 2), offs=offs).chunk(2, dim=-1)
        return torch._grouped_mm(act_fn(gate) * up, down_proj.transpose(1, 2), offs=offs)


class ReferenceExperts:
    """One ``F.linear`` pair per expert. Slow; for debugging and as the reference in tests."""

    def __call__(self, dispatched: DispatchOutput, gate_up_proj: Tensor, down_proj: Tensor, act_fn: ActFn) -> Tensor:
        counts = dispatched.tokens_per_expert_cpu
        if counts is None:
            counts = dispatched.tokens_per_expert.tolist()
        outputs = []
        for expert, rows in enumerate(dispatched.tokens.split(counts)):
            gate, up = F.linear(rows, gate_up_proj[expert]).chunk(2, dim=-1)
            outputs.append(F.linear(act_fn(gate) * up, down_proj[expert]))
        if not outputs:
            return dispatched.tokens.new_zeros(0, down_proj.size(1))
        return torch.cat(outputs)
