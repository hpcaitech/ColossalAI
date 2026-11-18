"""Counting the FLOPs a model runs: the matmuls through ``torch.utils.flop_counter.FlopCounterMode``, attention from
its shapes through ``AttentionCounter``.

Two ways to count, both used by the tool:

- on the meta device, before measuring (``metrics.compute.count_flops``): attention is replaced by a stand-in that
  computes nothing, so a model of any size is counted in seconds and without memory;
- on the devices, in one extra, untimed step of a measurement (``count_on_device``), for a model whose computation
  depends on values the meta device does not have: a MoE layer routes each token by its router's scores, and only a
  real step with real weights runs the experts it picks. Attention then runs for real and is counted on the side.

In both, attention is counted as its module declares it (causal, sliding window), not as the kernel's own FLOPs
formula would: a fused causal kernel skips half the score matrix, which ``FlopCounterMode``'s formulas do not know.
"""

import contextlib
from typing import Any, Callable, Mapping, Optional, Tuple
from unittest import mock

import torch
from torch.utils.flop_counter import FlopCounterMode, flop_registry


def attended_pairs(q_len: int, kv_len: int, causal: bool, window: Optional[int] = None) -> float:
    """Query-key pairs an attention computes: all of them; half for a causal mask; for a causal sliding window of
    ``window`` keys, ``q_len · window − window² / 2`` (the first ``window`` queries see fewer keys), which is half of
    the full matrix when the window covers the whole sequence."""
    if not causal:
        return q_len * kv_len
    if window is None or window >= kv_len:
        return q_len * kv_len / 2
    return q_len * window - window * window / 2


class _Attention(torch.autograd.Function):
    """Stands in for attention on the meta device: outputs and gradients of the right shapes, no computation."""

    @staticmethod
    def forward(ctx, counter, causal, window, query, key, value):
        ctx.counter, ctx.causal, ctx.window = counter, causal, window
        ctx.shapes = (query.shape, key.shape, value.shape)
        counter.add(*ctx.shapes, causal=causal, window=window, backward=False)
        return query.new_empty(*query.shape[:-1], value.shape[-1])

    @staticmethod
    def backward(ctx, grad):
        ctx.counter.add(*ctx.shapes, causal=ctx.causal, window=ctx.window, backward=True)
        return (None, None, None, *(grad.new_zeros(shape) for shape in ctx.shapes))


class AttentionCounter:
    """Adds up the FLOPs of attention's score and value products from each call's shapes:
    ``2 · batch · heads · pairs · (head_dim + v_dim)`` forward, twice that backward, where ``pairs`` is
    ``attended_pairs``.

    It replaces two entry points (``patched``):

    - ``hf_attention``, the ``"sdpa"`` attention of transformers (``AttentionInterface``), which is given the
      attention module: causality and the sliding window come from the module's own ``is_causal`` and
      ``sliding_window``, as transformers itself decides them, whatever mask tensor it passes;
    - ``sdpa``, ``torch.nn.functional.scaled_dot_product_attention`` called directly: only ``is_causal`` tells. A call
      with a mask tensor and ``is_causal=False`` is counted whole, as the mask's values are not looked at, and counted
      in ``unseen_masks``, so that the report can say the count may overstate it.

    With ``compute=False`` (the meta device) the calls are replaced by a stand-in; with ``compute=True`` (a real step)
    they run, and their backward is counted when the gradient reaches their output. Under gradient checkpointing the
    recomputed forward then counts once more and its backward does not: the hardware FLOPs."""

    def __init__(self, compute: bool = False):
        self.compute, self.flops, self.unseen_masks = compute, 0, 0
        self._hf_sdpa: Optional[Callable] = None
        # inside transformers' own sdpa, which calls scaled_dot_product_attention: counted once, by hf_attention
        self._inside_hf = False
        self._sdpa = torch.nn.functional.scaled_dot_product_attention

    def hf_attention(self, module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
        causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        window = kw.get("sliding_window") or getattr(module, "sliding_window", None)
        if not self.compute:
            # transformers returns (batch, q_len, heads, v_dim) and no weights
            return _Attention.apply(self, causal, window, query, key, value).transpose(1, 2), None
        extra = {} if is_causal is None else {"is_causal": is_causal}
        self._inside_hf = True
        try:
            output, weights = self._hf_sdpa(
                module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **extra, **kw
            )
        finally:
            self._inside_hf = False
        self._count_running(query.shape, key.shape, value.shape, causal, window, output)
        return output, weights

    def sdpa(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        if self._inside_hf:
            return self._sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs
            )
        if attn_mask is not None and not is_causal:
            self.unseen_masks += 1
        if not self.compute:
            return _Attention.apply(self, is_causal, None, query, key, value)
        output = self._sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
        self._count_running(query.shape, key.shape, value.shape, is_causal, None, output)
        return output

    def add(self, query, key, value, causal: bool, window: Optional[int], backward: bool):
        *batch, q_len, head_dim = query
        products = 2 * attended_pairs(q_len, key[-2], causal, window) * (head_dim + value[-1])
        for size in batch:
            products *= size
        self.flops += int(2 * products if backward else products)

    def _count_running(self, query, key, value, causal, window, output):
        self.add(query, key, value, causal, window, backward=False)
        if output.requires_grad:
            output.register_hook(lambda grad: self.add(query, key, value, causal, window, backward=True))

    def patched(self):
        """Replaces both entry points for every model while counting."""
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(torch.nn.functional, "scaled_dot_product_attention", self.sdpa))
        try:
            from transformers.modeling_utils import AttentionInterface
        except ImportError:
            return stack
        self._hf_sdpa = AttentionInterface._global_mapping["sdpa"]
        stack.enter_context(mock.patch.dict(AttentionInterface._global_mapping, {"sdpa": self.hf_attention}))
        return stack


def _zero(*args, **kwargs) -> int:
    return 0


# the attention kernels FlopCounterMode knows: counted by AttentionCounter instead, as the module declares them
_KERNEL_ATTENTION = {op: _zero for op in flop_registry if "attention" in str(op)}


def grouped_mm_flops(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    """FLOPs of ``torch._grouped_mm``, which FlopCounterMode does not know: one matmul per group, the groups
    splitting the rows (2-D x 3-D, as the experts of a MoE layer each take their tokens), the reduced dimension
    (2-D x 2-D, their weight gradients) or the columns (3-D x 2-D), or a batch of them (3-D x 3-D)."""
    if len(a_shape) == 2 and len(b_shape) == 3:
        return 2 * a_shape[0] * a_shape[1] * b_shape[2]
    if len(a_shape) == 2 and len(b_shape) == 2:
        return 2 * a_shape[0] * a_shape[1] * b_shape[1]
    if len(a_shape) == 3 and len(b_shape) == 3:
        return 2 * a_shape[0] * a_shape[1] * a_shape[2] * b_shape[2]
    return 2 * a_shape[1] * a_shape[2] * b_shape[1]


# operators FlopCounterMode does not know, counted for every model
BUILTIN_FORMULAS = {torch.ops.aten._grouped_mm: grouped_mm_flops} if hasattr(torch.ops.aten, "_grouped_mm") else {}


def count_on_device(run_step: Callable[[], Any], flop_formulas: Mapping[Any, Callable[..., int]]) -> Tuple[int, int]:
    """The FLOPs ``run_step()`` runs on this rank, and the attention calls whose masks were not looked at."""
    attention = AttentionCounter(compute=True)
    counter = FlopCounterMode(display=False, custom_mapping={**_KERNEL_ATTENTION, **BUILTIN_FORMULAS, **flop_formulas})
    with counter, attention.patched():
        run_step()
    return counter.get_total_flops() + attention.flops, attention.unseen_masks
