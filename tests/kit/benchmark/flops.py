"""FLOPs per trained token, the numerator of MFU.

Model FLOPs: the matrix multiplications of one forward and backward pass (PaLM, Chowdhery et al. 2022, appendix B),
without the forward that gradient checkpointing recomputes (counting that gives hardware FLOPs, HFU). Causal attention
is halved, as Megatron-LM and FlashAttention count it: a causal mask leaves half of the score matrix, and fused kernels
skip the other half. PaLM counts the whole matrix; at a 20k-token sequence that overstates a Llama-3.2-1B's FLOPs so
much that its measured MFU exceeds 100%.

Two independent sources:

- ``decoder_flops_per_token``: the closed formula for a HuggingFace decoder-only model with a gated MLP, from its
  config alone.
- ``counted_flops_per_token``: ``torch.utils.flop_counter.FlopCounterMode`` over one forward and backward of the
  model built on the meta device (as fake tensors), so no memory is used and any model can be counted. Calls to
  ``scaled_dot_product_attention`` are counted from their shapes instead (``AttentionCounter``), halved when causal:
  the counter cannot see a mask, and on the meta device the call would run as plain matmuls over the whole matrix.
  Attention written as plain matmuls (``attn_implementation="eager"``) is counted whole, as it is computed whole.

An entry with a formula reports both, and their difference is flagged above ``FLOPS_MISMATCH_LIMIT``. The counter
also sees the few matmuls that are not weights, e.g. a rotary embedding's frequency product (head_dim / 2 FLOPs per
token); for real models they are below 1e-6 of the total.
"""

from typing import Any, Dict, Optional
from unittest import mock

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.flop_counter import FlopCounterMode

# a formula and the counted FLOPs that differ by more than this disagree about the model: one of them is wrong
FLOPS_MISMATCH_LIMIT = 0.05


def decoder_flops_per_token(model: nn.Module, seq_len: int) -> float:
    """Model FLOPs per token of a decoder-only transformer with grouped-query attention and a gated MLP (Llama,
    Qwen3, Mistral), from ``model.config``.

    Per layer the weight matmuls are q, k, v, o and the three MLP projections; the output projection to the
    vocabulary is counted once (an input embedding is a lookup, not a matmul, also when tied to it). Each weight
    costs 6 FLOPs per token (2 forward, 4 backward). The attention scores and their product with the values cost
    ``2 · 2 · seq_len · heads · head_dim`` per token and layer forward, three times that with backward, halved for the
    causal mask."""
    c = model.config
    heads, kv_heads = c.num_attention_heads, c.num_key_value_heads
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // heads
    attention_weights = c.hidden_size * head_dim * (2 * heads + 2 * kv_heads)
    mlp_weights = 3 * c.hidden_size * c.intermediate_size
    matmul_weights = c.num_hidden_layers * (attention_weights + mlp_weights) + c.hidden_size * c.vocab_size
    attention = 12 * c.num_hidden_layers * seq_len * heads * head_dim // 2
    return 6 * matmul_weights + attention


def recompute_countable() -> bool:
    """Whether this torch can recompute a checkpointed forward on the meta device. Older versions look up the device's
    RNG module even when no RNG state is to be restored, and the meta device has none."""
    try:
        with torch.random.fork_rng(devices=[], enabled=False, device_type="meta"):
            return True
    except RuntimeError:
        return False


class _Attention(torch.autograd.Function):
    """Stands in for attention while counting: outputs and gradients of the right shapes, no computation."""

    @staticmethod
    def forward(ctx, counter, causal, query, key, value):
        ctx.counter, ctx.causal, ctx.shapes = counter, causal, (query.shape, key.shape, value.shape)
        counter.add(*ctx.shapes, causal=causal, backward=False)
        return query.new_empty(*query.shape[:-1], value.shape[-1])

    @staticmethod
    def backward(ctx, grad):
        ctx.counter.add(*ctx.shapes, causal=ctx.causal, backward=True)
        return (None, None, *(grad.new_zeros(shape) for shape in ctx.shapes))


class AttentionCounter:
    """Replaces ``torch.nn.functional.scaled_dot_product_attention`` while counting, and adds up the FLOPs of the
    score and value products from each call's shapes: ``2 · batch · heads · q_len · kv_len · (head_dim + v_dim)``
    forward, twice that backward, halved when the call or the model is causal (``is_causal=True``, or ``causal`` for
    a model that passes its causal mask as a tensor, whose values are not known on the meta device)."""

    def __init__(self, causal: bool):
        self.causal, self.flops = causal, 0

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        return _Attention.apply(self, is_causal or self.causal, query, key, value)

    def add(self, query, key, value, causal: bool, backward: bool):
        *batch, q_len, head_dim = query
        products = 2 * q_len * key[-2] * (head_dim + value[-1])
        for size in batch:
            products *= size
        if causal:
            products //= 2
        self.flops += 2 * products if backward else products


def counted_flops_per_token(spec, size: str, seq_len: int, micro_batch: int, recompute: bool = False) -> float:
    """FLOPs per token that ``FlopCounterMode`` counts over one forward and backward of ``spec.models[size]`` on the
    meta device. ``recompute=True`` enables non-reentrant gradient checkpointing, so that the recomputed forward is
    counted too (hardware FLOPs).

    The meta tensors are also fake tensors (``FakeTensorMode``): code that branches on tensor values only when it is
    not being traced, such as HuggingFace's check for packed sequences, then takes its traced path instead of reading
    a value that a meta tensor does not have."""
    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("meta"):
        model = spec.models[size]()
        if spec.prepare_fn is not None:
            model = spec.prepare_fn(model) or model
        if recompute:
            if not recompute_countable():
                raise NotImplementedError(
                    f"torch {torch.__version__} cannot recompute a checkpointed forward on the meta device "
                    "(torch.random.fork_rng needs a device module), so hardware FLOPs cannot be counted"
                )
            # counting does not depend on random numbers; the meta device has no RNG state to restore
            kwargs = {"use_reentrant": False, "preserve_rng_state": False}
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
        model.train()
        with torch.device("cpu"):
            batch = spec.batch_fn(model, micro_batch, seq_len, torch.Generator().manual_seed(0))
        tokens = spec.tokens_fn(batch)
        batch = {k: v.to("meta") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        counter, attention = FlopCounterMode(display=False), AttentionCounter(spec.causal_attention)
        with counter, mock.patch.object(torch.nn.functional, "scaled_dot_product_attention", attention):
            spec.loss_fn(model(**batch)).backward()
    return (counter.get_total_flops() + attention.flops) / tokens


def flops_per_token(spec, size: str, seq_len: int, micro_batch: int, gradient_checkpointing: bool) -> Dict[str, Any]:
    """What a report records about FLOPs: the model FLOPs per token used for MFU (the formula's when the entry has
    one, else the counted), both sources and their difference, and the hardware FLOPs per token for HFU."""
    counted = counted_flops_per_token(spec, size, seq_len, micro_batch)
    analytic: Optional[float] = None
    if spec.flops_fn is not None:
        with torch.device("meta"):
            analytic = spec.flops_fn(spec.models[size](), seq_len)
    model = analytic if analytic is not None else counted
    hardware = (
        counted_flops_per_token(spec, size, seq_len, micro_batch, recompute=True) if gradient_checkpointing else None
    )
    return {
        "model_flops_per_token": model,
        "source": "formula" if analytic is not None else "counted",
        "analytic": analytic,
        "counted": counted,
        "difference": None if analytic is None else (counted - analytic) / analytic,
        # with gradient checkpointing the recomputed forward is done too; without, hardware and model FLOPs are equal
        "hardware_flops_per_token": hardware if hardware is not None else model,
    }
