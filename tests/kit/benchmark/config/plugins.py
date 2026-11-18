"""The plugins and precisions a model is measured under, by name, including the built-in baseline without ColossalAI.

The baseline trains the same model with plain PyTorch FSDP2 (``fully_shard``), so that a report can say how a plugin
compares with what PyTorch gives out of the box. The model is sharded the way torchtitan does it: ``fully_shard`` on
every transformer block, then on the root, with a ``MixedPrecisionPolicy`` that computes in bf16 and reduces gradients
in float32 (the float32 parameters are the master weights). Transformer blocks are found by the class names a
HuggingFace model lists in ``_no_split_modules``.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecision

from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.booster.plugin.torch_fsdp_plugin import TorchFSDPPlugin


@dataclass(frozen=True)
class PluginConfig:
    """A plugin and precision to measure under. ``plugin_fn`` builds the plugin inside each process; ``dtype`` is what
    the matmuls run in, which selects the device's peak for MFU. A plain-PyTorch baseline has no plugin but a
    ``wrap_fn(model, device)`` that parallelizes the model itself.

    Only HybridParallelPlugin (and GeminiPlugin with tensor parallelism, not used here) replaces the model's modules,
    by a shardformer policy, when it boosts the model; DDP, LowLevelZeRO, FSDP and Gemini without tensor parallelism
    train the model as built."""

    plugin_fn: Optional[Callable]
    dtype: str
    mixed_precision: Optional[str] = None
    wrap_fn: Optional[Callable] = None
    # the plugin replaces modules by a shardformer policy, and takes the model's own (BenchmarkSpec.shardformer_policy)
    takes_policy: bool = False
    # the plugin runs the optimizer on the CPU (offload): AdamW becomes HybridAdam (optimizers.for_cpu)
    cpu_optimizer: bool = False


def _fsdp2():
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError:  # torch < 2.6
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    return fully_shard, MixedPrecisionPolicy


def transformer_blocks(model: nn.Module) -> List[nn.Module]:
    """The modules of ``model`` whose class is in its ``_no_split_modules``, outermost first and not nested in one
    another. A model that does not declare any is sharded as one unit."""
    names = set(getattr(model, "_no_split_modules", None) or ())
    blocks, inside = [], set()
    for name, module in model.named_modules():
        if any(name.startswith(prefix + ".") for prefix in inside):
            continue
        if type(module).__name__ in names:
            blocks.append(module)
            inside.add(name)
    return blocks


def fsdp2_bf16(model: nn.Module, device: torch.device) -> nn.Module:
    """Shard ``model`` (float32, on the CPU) over the default process group with FSDP2, bf16 compute and float32
    gradient reduction; it is moved to ``device`` first."""
    fully_shard, MixedPrecisionPolicy = _fsdp2()
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    model.to(device)
    for block in transformer_blocks(model):
        fully_shard(block, mp_policy=policy)
    fully_shard(model, mp_policy=policy)
    return model


FSDP_BF16 = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
# The names match the configurations of tests/kit/numerics/acceptance.py, so a model's alignment and benchmark reports
# refer to the same setups.
CONFIGS: Dict[str, PluginConfig] = {
    "ddp_fp32": PluginConfig(TorchDDPPlugin, "fp32"),
    # autocast: the matmuls run in fp16
    "ddp_fp16": PluginConfig(TorchDDPPlugin, "fp16", "fp16"),
    "zero1_bf16": PluginConfig(partial(LowLevelZeroPlugin, stage=1, precision="bf16"), "bf16"),
    "zero2_bf16": PluginConfig(partial(LowLevelZeroPlugin, stage=2, precision="bf16"), "bf16"),
    "zero1_bf16_offload": PluginConfig(
        partial(LowLevelZeroPlugin, stage=1, precision="bf16", cpu_offload=True), "bf16", cpu_optimizer=True
    ),
    "zero2_bf16_offload": PluginConfig(
        partial(LowLevelZeroPlugin, stage=2, precision="bf16", cpu_offload=True), "bf16", cpu_optimizer=True
    ),
    "gemini_bf16": PluginConfig(partial(GeminiPlugin, precision="bf16"), "bf16"),
    "fsdp_fp32": PluginConfig(TorchFSDPPlugin, "fp32"),
    "fsdp_bf16": PluginConfig(partial(TorchFSDPPlugin, mixed_precision=FSDP_BF16), "bf16"),
    # data parallel only, with the model's shardformer policy applied: for models whose training uses
    # HybridParallelPlugin (no tensor or pipeline parallelism)
    "hybrid_zero2_bf16": PluginConfig(
        partial(HybridParallelPlugin, tp_size=1, pp_size=1, zero_stage=2, precision="bf16"), "bf16", takes_policy=True
    ),
    # the built-in baseline, without ColossalAI
    "pytorch_fsdp2_bf16": PluginConfig(None, "bf16", wrap_fn=fsdp2_bf16),
}
