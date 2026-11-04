r"""ShardFormer policy for the diffusers Cosmos 3 omni transformer.

Cosmos 3 is a mixture-of-transformers diffusion policy: a Qwen-style text tower (the understanding / AR pathway) and
a diffusion tower (the generation / DM pathway) share one stack of decoder layers, and each layer's attention runs
both pathways through :class:`~colossalai.shardformer.modeling.cosmos3.ColossalTwoWayAttnProcessor`.

What the policy has to do
-------------------------
ShardFormer is the model-rewrite pass that every :class:`~colossalai.booster.plugin.HybridParallelPlugin` run goes
through, including pure ZeRO runs where no tensor parallelism is requested. Two things therefore belong here:

1. **The attention processor.** It is not a parameter, so neither ZeRO nor DDP touches it, and the model builds its
   own stock ``Cosmos3AttnProcessor`` in every ``Cosmos3PackedMoTAttention.__init__``. Installing ours is done in
   :meth:`Cosmos3OmniPolicy.postprocess`, i.e. after the module replacements and after lazy materialization, so the
   processor sees the final (possibly TP-sharded) modules and nothing rewrites it afterwards.
2. **Tensor-parallel rewrites**, mirroring the usual ShardFormer recipe (column-parallel q/k/v and gate/up, row-parallel
   output and down projections, local head counts on the attention module). The official cosmos-framework only ships
   FSDP2 (+ context parallelism, activation checkpointing, ``torch.compile``) and has no tensor-parallel plan, so this
   part is the ShardFormer equivalent rather than a port.

The official recipe shards per decoder block; ``HybridParallelPlugin`` reaches the same granularity because ZeRO
partitions the whole module's gradients and the policy rewrites inside each block.

Multi-sample packs
------------------
The per-sample AR/DM boundaries of a packed batch are runtime data, not a property of the module graph, so they are
set on the shared processor before each forward:

.. code-block:: python

    policy = Cosmos3OmniPolicy(backend="npu")
    model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, dataloader=dataloader)
    ...
    policy.set_packed_meta(layout.meta)
    outputs = model(**inputs)

:func:`~colossalai.shardformer.modeling.cosmos3.set_cosmos3_packed_meta` does the same from the model side when the
policy object is not at hand.
"""

from typing import Dict, Optional, Union

import torch.nn as nn

import colossalai.shardformer.layer as col_nn

from ..modeling.cosmos3 import ColossalTwoWayAttnProcessor
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["Cosmos3OmniPolicy"]

# The policy keys on class names rather than imported classes: the policy must stay importable without diffusers,
# and a diffusers install that predates Cosmos 3 would otherwise break `import colossalai.shardformer.policies.cosmos3`.
_ATTENTION_CLS = "Cosmos3PackedMoTAttention"
_MLP_CLS = "Cosmos3VLTextMLP"

# Projections that produce a head-split tensor and are therefore column-parallel, and the two that consume one and are
# row-parallel. The generation pathway has its own copy of all of them.
_COL_PROJECTIONS = ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")
_ROW_PROJECTIONS = ("to_out", "to_add_out")


class Cosmos3OmniPolicy(Policy):
    """ShardFormer policy for ``diffusers``' ``Cosmos3OmniTransformer``.

    Args:
        backend (str, optional): Attention backend for
            :class:`~colossalai.shardformer.modeling.cosmos3.ColossalTwoWayAttnProcessor`: ``"naive"`` (portable) or
            ``"npu"`` (Ascend fusion kernel). Defaults to ``"naive"``.
        processor (optional): A ready-made processor to install instead of building one from ``backend``. Use this to
            share a processor with code that already holds one, e.g. a script that pre-sets pack metadata.

    Raises:
        ValueError: If ``backend`` is not a known two-way kernel backend.
    """

    def __init__(self, backend: str = "naive", processor: Optional[ColossalTwoWayAttnProcessor] = None) -> None:
        super().__init__()
        self.processor = processor if processor is not None else ColossalTwoWayAttnProcessor(backend=backend)

    @property
    def packed_meta(self):
        """The pack metadata currently set on the installed processor, or ``None``."""
        return self.processor.packed_meta

    def set_packed_meta(self, packed_meta) -> None:
        """Set the per-sample AR/DM lengths used by the next forward.

        Args:
            packed_meta (Optional[TwoWayPackedMeta]): Metadata of the next pack, a
                :class:`~colossalai.shardformer.modeling.cosmos_mot.CosmosTwoWayLayout` carrying it, or ``None`` to fall
                back to "the whole pack is one sample".
        """
        self.processor.set_packed_meta(packed_meta)

    def config_sanity_check(self) -> None:
        if self.pipeline_stage_manager is not None:
            raise NotImplementedError(
                "Cosmos3OmniPolicy does not support pipeline parallelism: the omni transformer runs one shared "
                "decoder stack for both pathways, so a stage split has no layer boundary that keeps the two-way "
                "attention intact."
            )
        if self.shard_config.enable_sequence_parallelism:
            raise NotImplementedError(
                "Cosmos3OmniPolicy does not support sequence parallelism: the two-way flat-attention kernel consumes "
                "whole per-sample AR and DM blocks, which a sequence split would cut across."
            )
        unsupported = [
            flag
            for flag in ("enable_fused_normalization", "enable_flash_attention", "enable_jit_fused")
            if getattr(self.shard_config, flag, False)
        ]
        if unsupported:
            # The Cosmos 3 stack uses diffusers RMSNorm and its own processor-based attention, so none of ShardFormer's
            # fused-norm / flash-attention / JIT-fused rewrites apply; better to say so than to silently skip them.
            raise NotImplementedError(
                "Cosmos3OmniPolicy does not implement ShardFormer's " + ", ".join(unsupported) + "."
            )
        if not self.shard_config.enable_tensor_parallelism:
            return

        tp_size = self.shard_config.tensor_parallel_size
        num_heads, num_kv_heads = self._head_counts()
        if num_heads % tp_size != 0:
            raise ValueError(
                f"Cosmos3 attention heads ({num_heads}) must be divisible by the tensor parallel size ({tp_size})."
            )
        if num_kv_heads % tp_size != 0:
            raise ValueError(
                f"Cosmos3 key/value heads ({num_kv_heads}) must be divisible by the tensor parallel size ({tp_size})."
            )
        if num_kv_heads // tp_size < 1:
            raise ValueError(
                f"Cosmos3 key/value heads ({num_kv_heads}) must be at least the tensor parallel size ({tp_size})."
            )

    def preprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        policy = {}
        if not self.shard_config.enable_tensor_parallelism:
            # ZeRO and DDP need no module rewrite at all; the processor install in `postprocess` is the whole job.
            return policy

        tp_size = self.shard_config.tensor_parallel_size
        num_heads, num_kv_heads = self._head_counts()
        # `head_dim` and the per-head q/k norms stay replicated: every rank keeps whole heads, and the two-way kernel
        # runs the same per-head attention on its local head slice.
        policy[_ATTENTION_CLS] = ModulePolicyDescription(
            attribute_replacement={
                "num_attention_heads": num_heads // tp_size,
                "num_key_value_heads": num_kv_heads // tp_size,
            },
            sub_module_replacement=[
                *[
                    SubModuleReplacementDescription(
                        suffix=suffix,
                        target_module=col_nn.Linear1D_Col,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication},
                    )
                    for suffix in _COL_PROJECTIONS
                ],
                *[
                    SubModuleReplacementDescription(
                        suffix=suffix,
                        target_module=col_nn.Linear1D_Row,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication},
                    )
                    for suffix in _ROW_PROJECTIONS
                ],
            ],
        )

        # `gate_proj` only exists for the SwiGLU variants; the `relu2` variant has `up_proj`/`down_proj` alone.
        policy[_MLP_CLS] = ModulePolicyDescription(
            sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="gate_proj",
                    target_module=col_nn.Linear1D_Col,
                    kwargs={"fp8_communication": self.shard_config.fp8_communication},
                    ignore_if_not_exist=True,
                ),
                SubModuleReplacementDescription(
                    suffix="up_proj",
                    target_module=col_nn.Linear1D_Col,
                    kwargs={"fp8_communication": self.shard_config.fp8_communication},
                ),
                SubModuleReplacementDescription(
                    suffix="down_proj",
                    target_module=col_nn.Linear1D_Row,
                    kwargs={"fp8_communication": self.shard_config.fp8_communication},
                ),
            ]
        )
        return policy

    def postprocess(self) -> nn.Module:
        # Runs after `_replace_module` and `_materialize`, so this is the last word on the processor: a replacement
        # pass later in the shard would drop it, which is exactly the `set_processor` timing this policy exists to
        # get right.
        installed = 0
        for module in self.model.modules():
            if module.__class__.__name__ != _ATTENTION_CLS:
                continue
            module.set_processor(self.processor)
            installed += 1
        if installed == 0:
            raise RuntimeError(
                f"Cosmos3OmniPolicy found no {_ATTENTION_CLS} module in {type(self.model).__name__}. The policy only "
                "applies to a Cosmos 3 omni transformer; check the model passed to the booster."
            )
        return self.model

    def _head_counts(self):
        """Return ``(num_attention_heads, num_key_value_heads)`` of the model, from its config when present."""
        config = getattr(self.model, "config", None)
        num_heads = getattr(config, "num_attention_heads", None)
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None:
            return num_heads, num_kv_heads
        for module in self.model.modules():
            if module.__class__.__name__ == _ATTENTION_CLS:
                return module.num_attention_heads, module.num_key_value_heads
        raise RuntimeError(
            f"Cannot read the Cosmos 3 head counts: neither `config.num_attention_heads` nor a {_ATTENTION_CLS} module "
            "was found."
        )
