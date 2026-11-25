import warnings
from typing import Dict, List, Union

import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
)

from colossalai.shardformer.layer import (
    FusedRMSNorm,
    Linear1D_Col,
    Linear1D_Row,
    PaddingEmbedding,
    VocabParallelEmbedding1D,
)

from ..modeling.qwen3_moe import Qwen3MoeEPExperts
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["Qwen3MoePolicy", "Qwen3MoeModelPolicy", "Qwen3MoeForCausalLMPolicy"]


class Qwen3MoePolicy(Policy):
    r"""Sharding policy of the transformers v5 Qwen3-MoE.

    - Attention, the dense ``Qwen3MoeMLP`` layers (``mlp_only_layers``), the embedding and the LM head follow the dense
      Qwen3 policy.
    - The router (``mlp.gate``) and the sparse block itself stay the native HF modules, so that v5's
      ``OutputRecorder(Qwen3MoeTopKRouter)`` still records the router logits.
    - With ``shard_config.ep_group`` the fused experts of every sparse block become a :class:`~colossalai.moe.EPMoELayer`.

    Not supported yet: pipeline parallelism, sequence parallelism, and expert parallelism together with tensor
    parallelism (the shared layer has no tensor-parallel experts).
    """

    def __init__(self) -> None:
        super().__init__()
        import transformers
        from packaging.version import Version

        assert Version(transformers.__version__) >= Version(
            "5.17.0"
        ), "The Qwen3-MoE model should run on a transformers version of 5.17.0 or higher."

    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.tie_weight = self.tie_weight_check()
        self.origin_attn_implement = self.model.config._attn_implementation
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        shard_config = self.shard_config
        if self.pipeline_stage_manager is not None:
            raise NotImplementedError("Pipeline parallelism is not supported for Qwen3-MoE yet.")
        if shard_config.enable_sequence_parallelism:
            raise NotImplementedError("Sequence parallelism is not supported for Qwen3-MoE yet.")
        tp_size = shard_config.tensor_parallel_size if shard_config.enable_tensor_parallelism else 1
        if shard_config.ep_group and tp_size > 1:
            raise NotImplementedError(
                "Expert parallelism together with tensor parallelism is not supported for Qwen3-MoE yet: "
                "the shared EPMoELayer has no tensor-parallel experts."
            )

        policy = {}
        config = self.model.config

        embedding_cls = None
        if shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        elif self.tie_weight:
            embedding_cls = PaddingEmbedding

        if shard_config.enable_tensor_parallelism:
            assert (
                config.num_attention_heads % tp_size == 0
            ), "The number of attention heads must be divisible by tensor parallel size."
            assert (
                config.num_key_value_heads % tp_size == 0
            ), "The number of key_value heads must be divisible by tensor parallel size."
            fp8 = shard_config.fp8_communication
            # no attribute replacement: transformers v5 attention infers the local number of heads from the shape of
            # its projections (`view(..., -1, head_dim)`) and does not read `num_heads` / `hidden_size`
            policy[Qwen3MoeDecoderLayer] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj", target_module=Linear1D_Col, kwargs=dict(fp8_communication=fp8)
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj", target_module=Linear1D_Col, kwargs=dict(fp8_communication=fp8)
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj", target_module=Linear1D_Col, kwargs=dict(fp8_communication=fp8)
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj", target_module=Linear1D_Row, kwargs=dict(fp8_communication=fp8)
                    ),
                    # The dense Qwen3MoeMLP of `mlp_only_layers`. Sparse layers have no such submodules, so skip them
                    # there; their experts are replicated over the tensor parallel group.
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(fp8_communication=fp8),
                        ignore_if_not_exist=True,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(fp8_communication=fp8),
                        ignore_if_not_exist=True,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=Linear1D_Row,
                        kwargs=dict(fp8_communication=fp8),
                        ignore_if_not_exist=True,
                    ),
                ],
            )

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=embedding_cls,
                    kwargs=(
                        {
                            "make_vocab_size_divisible_by": shard_config.make_vocab_size_divisible_by,
                            "fp8_communication": shard_config.fp8_communication,
                        }
                        if shard_config.enable_tensor_parallelism
                        else {"make_vocab_size_divisible_by": shard_config.make_vocab_size_divisible_by}
                    ),
                ),
                policy=policy,
                target_key=Qwen3MoeModel,
            )

        if shard_config.ep_group:
            # Only the experts are replaced: dense layers have no `mlp.experts` and are skipped, and the router keeps
            # its native class for the router-logits hook of transformers v5.
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="mlp.experts",
                    target_module=Qwen3MoeEPExperts,
                    kwargs={"ep_group": shard_config.ep_group},
                    ignore_if_not_exist=True,
                ),
                policy=policy,
                target_key=Qwen3MoeDecoderLayer,
            )

        if shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(suffix="input_layernorm", target_module=FusedRMSNorm),
                    SubModuleReplacementDescription(suffix="post_attention_layernorm", target_module=FusedRMSNorm),
                ],
                policy=policy,
                target_key=Qwen3MoeDecoderLayer,
            )
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(suffix="norm", target_module=FusedRMSNorm),
                policy=policy,
                target_key=Qwen3MoeModel,
            )

        if shard_config.enable_flash_attention:
            warnings.warn("Flash attention is natively supported in transformers, will ignore the flag.")
            shard_config.enable_flash_attention = False

        return policy

    def postprocess(self):
        if self.shard_config.enable_tensor_parallelism and self.shard_config.tensor_parallel_size > 1:
            tp_group = self.shard_config.tensor_parallel_process_group

            def sync_head_grad(grad):
                # Q/K norm weights are shared by the locally partitioned heads.
                grad = grad.clone()
                dist.all_reduce(grad, group=tp_group)
                return grad

            for module in self.model.modules():
                if isinstance(module, Qwen3MoeAttention):
                    for norm in (module.q_norm, module.k_norm):
                        weight = norm.weight
                        if (
                            weight is not None
                            and weight.requires_grad
                            and not getattr(weight, "_qwen3_moe_tp_grad_hook", False)
                        ):
                            weight.register_hook(sync_head_grad)
                            weight._qwen3_moe_tp_grad_hook = True
        return self.model

    def get_held_layers(self) -> List[Module]:
        raise NotImplementedError("Pipeline parallelism is not supported for Qwen3-MoE yet.")

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        return []


class Qwen3MoeModelPolicy(Qwen3MoePolicy):
    pass


class Qwen3MoeForCausalLMPolicy(Qwen3MoePolicy):
    def module_policy(self):
        policy = super().module_policy()
        if self.shard_config.enable_tensor_parallelism:
            policy[Qwen3MoeForCausalLM] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="lm_head",
                        target_module=Linear1D_Col,
                        kwargs=dict(gather_output=True, fp8_communication=self.shard_config.fp8_communication),
                    )
                ],
            )
        return policy
