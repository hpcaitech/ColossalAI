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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.flop_counter import FlopCounterMode

# the counter itself is the runner's, which also counts on the devices
from ..runner.counting import BUILTIN_FORMULAS, AttentionCounter

# a formula and the counted FLOPs that differ by more than this disagree about the model: one of them is wrong
FLOPS_MISMATCH_LIMIT = 0.05
# where the model FLOPs per token come from
FORMULA, COUNTED = "formula", "counted"
# where the count ran: on the meta device before measuring, or on the devices in one extra step of the measurement
META, DEVICE = "meta", "device"


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


def moe_decoder_flops_per_token(model: nn.Module, seq_len: int) -> float:
    """Model FLOPs per token of a decoder-only MoE transformer like Qwen3-MoE, from ``model.config``: as
    ``decoder_flops_per_token``, but in a sparse layer each token goes through its ``num_experts_per_tok`` experts
    (three projections of ``moe_intermediate_size`` each) and the router (``hidden_size × num_experts``), not through a
    dense MLP. Which experts changes from step to step; how many does not, so neither do the FLOPs. A layer is sparse
    unless it is in ``mlp_only_layers`` or ``decoder_sparse_step`` skips it."""
    c = model.config
    heads, kv_heads = c.num_attention_heads, c.num_key_value_heads
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // heads
    attention_weights = c.hidden_size * head_dim * (2 * heads + 2 * kv_heads)
    dense_mlp = 3 * c.hidden_size * c.intermediate_size
    sparse_mlp = c.num_experts_per_tok * 3 * c.hidden_size * c.moe_intermediate_size + c.hidden_size * c.num_experts
    step = getattr(c, "decoder_sparse_step", 1) or 1
    dense_only = set(getattr(c, "mlp_only_layers", None) or ())
    mlp_weights = sum(
        sparse_mlp if i not in dense_only and (i + 1) % step == 0 else dense_mlp for i in range(c.num_hidden_layers)
    )
    matmul_weights = c.num_hidden_layers * attention_weights + mlp_weights + c.hidden_size * c.vocab_size
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


def count_flops(spec, size: str, seq_len: int, micro_batch: int, recompute: bool = False) -> Dict[str, Any]:
    """FLOPs per token that ``FlopCounterMode`` counts over one forward and backward of the model at ``size`` in its
    reference implementation (``BenchmarkSpec.reference_implementation``), on the meta device, and how many attention
    calls had masks it could not see. ``recompute=True`` enables non-reentrant gradient checkpointing, so that the
    recomputed forward is counted too (hardware FLOPs).

    The meta tensors are also fake tensors (``FakeTensorMode``): code that branches on tensor values only when it is
    not being traced, such as HuggingFace's check for packed sequences, then takes its traced path instead of reading
    a value that a meta tensor does not have. Custom operators count through ``BenchmarkSpec.flop_formulas``."""
    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("meta"):
        model = spec.build(size, spec.reference_implementation, pretrained=False)
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
        counter = FlopCounterMode(display=False, custom_mapping={**BUILTIN_FORMULAS, **spec.flop_formulas})
        attention = AttentionCounter()
        with counter, attention.patched():
            spec.loss_fn(model, batch).backward()
    return {"per_token": (counter.get_total_flops() + attention.flops) / tokens, "unseen_masks": attention.unseen_masks}


def counted_flops_per_token(spec, size: str, seq_len: int, micro_batch: int, recompute: bool = False) -> float:
    """``count_flops``'s FLOPs per token."""
    return count_flops(spec, size, seq_len, micro_batch, recompute)["per_token"]


def flops_per_token(
    spec,
    size: str,
    seq_len: int,
    micro_batch: int,
    gradient_checkpointing: bool,
    device_count: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """What a report records about FLOPs: the model FLOPs per token used for MFU (the formula's when the entry has
    one, else the counted), both sources and their difference, attention masks the count could not see, and the
    hardware FLOPs per token for HFU.

    ``device_count`` (``{"per_token", "unseen_masks"}``) is a count made on the devices during the measurement
    (``BenchmarkSpec.count_on_device``), used instead of the meta device's. Under gradient checkpointing it includes
    the recomputed forward: it is then the hardware FLOPs, and there is no counted model FLOPs to check the formula
    against."""
    analytic: Optional[float] = None
    if spec.flops_fn is not None:
        with torch.device("meta"):
            analytic = spec.flops_fn(spec.build(size, spec.reference_implementation, pretrained=False), seq_len)
    if device_count is not None:
        count = device_count
        counted = None if gradient_checkpointing else count["per_token"]
        hardware = count["per_token"] if gradient_checkpointing else None
        where = DEVICE
    else:
        count = count_flops(spec, size, seq_len, micro_batch)
        counted = count["per_token"]
        hardware = (
            counted_flops_per_token(spec, size, seq_len, micro_batch, recompute=True)
            if gradient_checkpointing
            else None
        )
        where = META
    if analytic is None and counted is None:
        raise ValueError(
            "no FLOPs formula and no count of the model FLOPs: give a formula to measure with checkpointing"
        )
    model = analytic if analytic is not None else counted
    return {
        "model_flops_per_token": model,
        "source": FORMULA if analytic is not None else COUNTED,
        "analytic": analytic,
        "counted": counted,
        "difference": None if analytic is None or counted is None else (counted - analytic) / analytic,
        "counted_implementation": spec.reference_implementation if where == META else None,
        "counted_on": where,
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
