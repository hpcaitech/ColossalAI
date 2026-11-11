"""FLOPs per trained token, the numerator of MFU.

Model FLOPs follow PaLM (Chowdhery et al. 2022, appendix B): the matrix multiplications of one forward and backward
pass, without the forward that gradient checkpointing recomputes (counting that gives hardware FLOPs, HFU). Causal
attention is not halved: the score and value products are counted over the whole sequence, as PaLM and torchtitan do,
so numbers compare with theirs.

Two independent sources:

- ``decoder_flops_per_token``: the closed formula for a HuggingFace decoder-only model with a gated MLP, from its
  config alone.
- ``counted_flops_per_token``: ``torch.utils.flop_counter.FlopCounterMode`` over one forward and backward of the
  model built on the meta device (as fake tensors), so no memory is used and any model can be counted. On the meta device
  ``scaled_dot_product_attention`` runs as two batched matmuls, i.e. model FLOPs; a fused attention kernel on a real
  device would also count the products its backward recomputes.

An entry with a formula reports both, and their difference is flagged above ``FLOPS_MISMATCH_LIMIT``. The counter
also sees the few matmuls that are not weights, e.g. a rotary embedding's frequency product (head_dim / 2 FLOPs per
token); for real models they are below 1e-6 of the total.
"""

from typing import Any, Dict, Optional

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
    ``2 · 2 · seq_len · heads · head_dim`` per token and layer forward, three times that with backward."""
    c = model.config
    heads, kv_heads = c.num_attention_heads, c.num_key_value_heads
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // heads
    attention_weights = c.hidden_size * head_dim * (2 * heads + 2 * kv_heads)
    mlp_weights = 3 * c.hidden_size * c.intermediate_size
    matmul_weights = c.num_hidden_layers * (attention_weights + mlp_weights) + c.hidden_size * c.vocab_size
    attention = 12 * c.num_hidden_layers * seq_len * heads * head_dim
    return 6 * matmul_weights + attention


def recompute_countable() -> bool:
    """Whether this torch can recompute a checkpointed forward on the meta device. Older versions look up the device's
    RNG module even when no RNG state is to be restored, and the meta device has none."""
    try:
        with torch.random.fork_rng(devices=[], enabled=False, device_type="meta"):
            return True
    except RuntimeError:
        return False


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
        counter = FlopCounterMode(display=False)
        with counter:
            spec.loss_fn(model(**batch)).backward()
    return counter.get_total_flops() / tokens


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
