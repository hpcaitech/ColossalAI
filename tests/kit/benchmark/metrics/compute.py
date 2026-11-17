"""Compute: FLOPs per token (the numerator of MFU), the devices' peaks (the denominator), and MFU / HFU.

FLOPs
-----
Model FLOPs: the matrix multiplications of one forward and backward pass (PaLM, Chowdhery et al. 2022, appendix B),
without the forward that gradient checkpointing recomputes (counting that gives hardware FLOPs, HFU). Causal attention
is halved, as Megatron-LM and FlashAttention count it: a causal mask leaves half of the score matrix, and fused kernels
skip the other half. PaLM counts the whole matrix; at a 20k-token sequence that overstates a Llama-3.2-1B's FLOPs so
much that its measured MFU exceeds 100%.

Two independent sources:

- ``decoder_flops_per_token``: the closed formula for a HuggingFace decoder-only model with a gated MLP, from its
  config alone.
- ``counted_flops_per_token``: ``torch.utils.flop_counter.FlopCounterMode`` over one forward and backward of the
  model built on the meta device (as fake tensors), so no memory is used. Attention is counted from its shapes instead
  (``AttentionCounter``): halved when causal and limited to the window when sliding, as transformers' attention
  modules declare it; a direct ``scaled_dot_product_attention`` call with a mask tensor is counted whole and reported.
  Attention written as plain matmuls (``attn_implementation="eager"``) is counted whole, as it is computed whole.

The formula is what MFU uses; the count checks it, and a difference above ``FLOPS_MISMATCH_LIMIT`` is flagged. The
count sees the matmuls the reference implementation runs on the meta device, so it misses custom kernels without a
registered formula (``BenchmarkSpec.flop_formulas``) and cannot route a MoE layer (which needs values). The counter
also sees the few matmuls that are not weights, e.g. a rotary embedding's frequency product (head_dim / 2 FLOPs per
token); for real models they are below 1e-6 of the total.

A model with several implementations (``BenchmarkSpec.implementations``) is counted in its reference
implementation: fused kernels and communicating layers cannot run on the meta device, and every implementation
computes the same FLOPs.

Peaks
-----
Peak dense matmul throughput of each device.

Two numbers per device and dtype, reported side by side:

- ``nominal``: the vendor's dense (no sparsity) figure, the industry's denominator, so MFU compares with published
  numbers. Each entry names its source.
- ``measured``: the best median of square matmuls (n = 4096, 8192, 16384, device-event timed, TF32 / HF32 off for
  fp32) on our machines. A device that cannot sustain its nominal figure (the Ascend 910B2 is power-capped at 400 W
  and clocks down under load) is far from 100% MFU even when its matmuls are as fast as they get; MFU against the
  measured peak separates that from the framework's own overhead.

Devices are recognized by ``accelerator.get_device_name()``. A device or dtype missing here gives no MFU, which the
report says, rather than a guessed one.
"""

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from unittest import mock

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.flop_counter import FlopCounterMode

# a formula and the counted FLOPs that differ by more than this disagree about the model: one of them is wrong
FLOPS_MISMATCH_LIMIT = 0.05
# where the model FLOPs per token come from
FORMULA, COUNTED = "formula", "counted"


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
    def forward(ctx, counter, causal, window, query, key, value):
        ctx.counter, ctx.causal, ctx.window = counter, causal, window
        ctx.shapes = (query.shape, key.shape, value.shape)
        counter.add(*ctx.shapes, causal=causal, window=window, backward=False)
        return query.new_empty(*query.shape[:-1], value.shape[-1])

    @staticmethod
    def backward(ctx, grad):
        ctx.counter.add(*ctx.shapes, causal=ctx.causal, window=ctx.window, backward=True)
        return (None, None, None, *(grad.new_zeros(shape) for shape in ctx.shapes))


def attended_pairs(q_len: int, kv_len: int, causal: bool, window: Optional[int] = None) -> float:
    """Query-key pairs an attention computes: all of them; half for a causal mask; for a causal sliding window of
    ``window`` keys, ``q_len · window − window² / 2`` (the first ``window`` queries see fewer keys), which is half of
    the full matrix when the window covers the whole sequence."""
    if not causal:
        return q_len * kv_len
    if window is None or window >= kv_len:
        return q_len * kv_len / 2
    return q_len * window - window * window / 2


class AttentionCounter:
    """Stands in for attention while counting, and adds up the FLOPs of the score and value products from each call's
    shapes: ``2 · batch · heads · pairs · (head_dim + v_dim)`` forward, twice that backward, where ``pairs`` is
    ``attended_pairs``.

    It replaces two entry points:

    - ``hf_attention``, the ``"sdpa"`` attention of transformers (``AttentionInterface``), which is given the
      attention module: causality and the sliding window come from the module's own ``is_causal`` and
      ``sliding_window``, as transformers itself decides them, whatever mask tensor it passes;
    - ``sdpa``, ``torch.nn.functional.scaled_dot_product_attention`` called directly: only ``is_causal`` tells. A call
      with a mask tensor and ``is_causal=False`` is counted whole, as the mask's values are not known on the meta
      device, and counted in ``unseen_masks``, so that the report can say the count may overstate it."""

    def __init__(self):
        self.flops, self.unseen_masks = 0, 0

    def hf_attention(self, module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
        causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        window = kw.get("sliding_window") or getattr(module, "sliding_window", None)
        # transformers returns (batch, q_len, heads, v_dim) and no weights
        return _Attention.apply(self, causal, window, query, key, value).transpose(1, 2), None

    def sdpa(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        if attn_mask is not None and not is_causal:
            self.unseen_masks += 1
        return _Attention.apply(self, is_causal, None, query, key, value)

    def add(self, query, key, value, causal: bool, window: Optional[int], backward: bool):
        *batch, q_len, head_dim = query
        products = 2 * attended_pairs(q_len, key[-2], causal, window) * (head_dim + value[-1])
        for size in batch:
            products *= size
        self.flops += int(2 * products if backward else products)


def _hf_attention_patch(counter: AttentionCounter):
    """Replaces transformers' ``"sdpa"`` attention for every model while counting; nothing without transformers."""
    try:
        from transformers.modeling_utils import AttentionInterface
    except ImportError:
        return contextlib.nullcontext()
    return mock.patch.dict(AttentionInterface._global_mapping, {"sdpa": counter.hf_attention})


def count_flops(spec, size: str, seq_len: int, micro_batch: int, recompute: bool = False) -> Dict[str, Any]:
    """FLOPs per token that ``FlopCounterMode`` counts over one forward and backward of the model at ``size`` in its
    reference implementation (``BenchmarkSpec.reference_implementation``), on the meta device, and how many attention
    calls had masks it could not see. ``recompute=True`` enables non-reentrant gradient checkpointing, so that the
    recomputed forward is counted too (hardware FLOPs).

    The meta tensors are also fake tensors (``FakeTensorMode``): code that branches on tensor values only when it is
    not being traced, such as HuggingFace's check for packed sequences, then takes its traced path instead of reading
    a value that a meta tensor does not have. Custom operators count through ``BenchmarkSpec.flop_formulas``."""
    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("meta"):
        model = spec.build(size, spec.reference_implementation)
        if recompute:
            if not recompute_countable():
                raise NotImplementedError(
                    f"torch {torch.__version__} cannot recompute a checkpointed forward on the meta device "
                    "(torch.random.fork_rng needs a device module), so hardware FLOPs cannot be counted"
                )
            # counting does not depend on random numbers; the meta device has no RNG state to restore
            spec.checkpointing_fn(model, False)
        model.train()
        with torch.device("cpu"):
            batch = spec.batch_fn(model, micro_batch, seq_len, torch.Generator().manual_seed(0))
        tokens = spec.tokens_fn(batch)
        batch = {k: v.to("meta") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        counter = FlopCounterMode(display=False, custom_mapping=dict(spec.flop_formulas) or None)
        attention = AttentionCounter()
        with (
            counter,
            _hf_attention_patch(attention),
            mock.patch.object(torch.nn.functional, "scaled_dot_product_attention", attention.sdpa),
        ):
            spec.loss_fn(model, batch).backward()
    return {"per_token": (counter.get_total_flops() + attention.flops) / tokens, "unseen_masks": attention.unseen_masks}


def counted_flops_per_token(spec, size: str, seq_len: int, micro_batch: int, recompute: bool = False) -> float:
    """``count_flops``'s FLOPs per token."""
    return count_flops(spec, size, seq_len, micro_batch, recompute)["per_token"]


def flops_per_token(spec, size: str, seq_len: int, micro_batch: int, gradient_checkpointing: bool) -> Dict[str, Any]:
    """What a report records about FLOPs: the model FLOPs per token used for MFU (the formula's when the entry has
    one, else the counted), both sources and their difference, attention masks the count could not see, and the
    hardware FLOPs per token for HFU."""
    count = count_flops(spec, size, seq_len, micro_batch)
    counted = count["per_token"]
    analytic: Optional[float] = None
    if spec.flops_fn is not None:
        with torch.device("meta"):
            analytic = spec.flops_fn(spec.build(size, spec.reference_implementation), seq_len)
    model = analytic if analytic is not None else counted
    hardware = (
        counted_flops_per_token(spec, size, seq_len, micro_batch, recompute=True) if gradient_checkpointing else None
    )
    return {
        "model_flops_per_token": model,
        "source": FORMULA if analytic is not None else COUNTED,
        "analytic": analytic,
        "counted": counted,
        "difference": None if analytic is None else (counted - analytic) / analytic,
        "counted_implementation": spec.reference_implementation,
        "unseen_masks": count["unseen_masks"],
        # with gradient checkpointing the recomputed forward is done too; without, hardware and model FLOPs are equal
        "hardware_flops_per_token": hardware if hardware is not None else model,
    }


@dataclass(frozen=True)
class Peak:
    device: str
    dtype: str
    nominal_tflops: Optional[float]
    nominal_source: str
    measured_tflops: Optional[float] = None
    measured_source: str = ""


_H20_NOMINAL = (
    "NVIDIA has no public H20 datasheet; figure consistent across third-party listings and press (dense Tensor Core)"
)
_H20_MEASURED = "one H20 (driver 595.71.05, SM clock steady at 1980 MHz, ~488 W of 500 W), torch 2.11, 2026-11-11"
_910B2_NOMINAL = (
    "Huawei, Atlas 800T A2 training server user guide (version 20, 2025-11-17), table 3-1, peak dense compute "
    "provided stably: {figures} depending on the processor; the higher is the 910B2 (24 cube cores against the "
    "910B3's 20 at the same 1800 MHz, CANN platform config)"
)
_910B2_MEASURED = (
    "two 910B2 servers (CANN 9.1.0, torch_npu 2.13.0rc1), 2026-11-11; power at the 400 W cap, AI core clock down "
    "from 1800 to 1400-1600 MHz"
)
_910B2_FP16 = _910B2_NOMINAL.format(figures="313 or 376 TFLOPS@FP16")

PEAKS: Tuple[Peak, ...] = (
    Peak("H20", "bf16", 148.0, _H20_NOMINAL, 140.6, _H20_MEASURED),
    Peak("H20", "fp16", 148.0, _H20_NOMINAL, 140.5, _H20_MEASURED),
    Peak("H20", "fp32", 44.0, _H20_NOMINAL + "; FP32 without Tensor Cores", 32.5, _H20_MEASURED),
    Peak("Ascend910B2", "fp16", 376.0, _910B2_FP16, 260.5, _910B2_MEASURED),
    Peak(
        "Ascend910B2", "bf16", 376.0, _910B2_FP16 + "; BF16 is not listed, taken equal to FP16", 290.8, _910B2_MEASURED
    ),
    Peak("Ascend910B2", "fp32", 99.0, _910B2_NOMINAL.format(figures="82 or 99 TFLOPS@FP32"), 75.3, _910B2_MEASURED),
)


def device_key(device_name: str) -> Optional[str]:
    """The ``PEAKS`` device that a ``get_device_name()`` string names, or None.

    ``NVIDIA H20`` and its larger-memory variant have the same compute; Ascend names are matched exactly, as the
    910B variants differ in compute."""
    if "H20" in device_name.replace("-", " ").split():
        return "H20"
    if device_name in {p.device for p in PEAKS}:
        return device_name
    return None


def lookup(device_name: str, dtype: str) -> Optional[Peak]:
    key = device_key(device_name)
    return next((p for p in PEAKS if p.device == key and p.dtype == dtype), None)


def utilization(tokens_per_s_per_device: float, flops: Dict[str, Any], peak: Dict[str, Any]) -> Dict[str, Any]:
    """Model and hardware TFLOPS per device at ``tokens_per_s_per_device``, and MFU / HFU against the nominal and the
    measured peak. ``flops`` is ``flops_per_token(...)``; ``peak`` a ``Peak`` as a dict, whose TFLOPS are None when the
    device's peak is not known (no MFU then)."""
    result = {
        "model_tflops_per_device": flops["model_flops_per_token"] * tokens_per_s_per_device / 1e12,
        "hardware_tflops_per_device": flops["hardware_flops_per_token"] * tokens_per_s_per_device / 1e12,
    }
    for name, tflops in (("mfu", result["model_tflops_per_device"]), ("hfu", result["hardware_tflops_per_device"])):
        for kind in ("nominal", "measured"):
            denominator = peak[f"{kind}_tflops"]
            result[f"{name}_{kind}"] = tflops / denominator if denominator else None
    return result
