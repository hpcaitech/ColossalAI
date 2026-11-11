"""The built-in baseline: the same model trained with plain PyTorch FSDP2 (``fully_shard``), no ColossalAI, timed by the
same protocol, so that a report can say how a plugin compares with what PyTorch gives out of the box.

The model is sharded the way torchtitan does it: ``fully_shard`` on every transformer block, then on the root, with a
``MixedPrecisionPolicy`` that computes in bf16 and reduces gradients in float32 (the float32 parameters are the master
weights). Transformer blocks are found by the class names a HuggingFace model lists in ``_no_split_modules``.
"""

from typing import List

import torch
from torch import nn


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
