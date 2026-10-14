# Modified from colossalai/shardformer/policies/mixtral.py and colossalai/shardformer/policies/qwen3.py
from functools import partial
from typing import Callable, Dict, List, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from colossalai.shardformer.layer import (
    FusedRMSNorm,
    Linear1D_Col,
    Linear1D_Row,
    PaddingEmbedding,
    RMSNorm,
    VocabParallelEmbedding1D,
)
from colossalai.shardformer.modeling.qwen3 import get_qwen3_flash_attention_forward
from colossalai.shardformer.modeling.qwen3_moe import EPQwen3MoeSparseMoeBlock, Qwen3MoePipelineForwards
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["Qwen3MoePolicy", "Qwen3MoeModelPolicy", "Qwen3MoeForCausalLMPolicy"]


class Qwen3MoePolicy(Policy):
    def __init__(self) -> None:
        super().__init__()
        import transformers
        from packaging.version import Version

        assert Version(transformers.__version__) >= Version(
            "4.51.0"
        ), "The Qwen3-MoE model should run on a transformers version of 4.51.0 or higher."

    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.tie_weight = self.tie_weight_check()
        self.origin_attn_implement = self.model.config._attn_implementation
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeAttention,
            Qwen3MoeDecoderLayer,
            Qwen3MoeModel,
        )

        policy = {}

        sp_mode = self.shard_config.sequence_parallelism_mode or None
        sp_size = self.shard_config.sequence_parallel_size or None
        sp_group = self.shard_config.sequence_parallel_process_group or None
        tp_size = self.shard_config.tensor_parallel_size
        if self.shard_config.enable_sequence_parallelism:
            if sp_mode != "all_to_all":
                raise NotImplementedError(
                    f"Sequence parallelism mode {sp_mode} is not supported for Qwen3-MoE yet, please use all_to_all."
                )
            if self.pipeline_stage_manager is not None:
                raise NotImplementedError("Sequence parallelism is not supported with pipeline parallelism.")
        if self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv:
            raise NotImplementedError("The zero bubble pipeline schedule is not supported for Qwen3-MoE yet.")

        norm_cls = FusedRMSNorm if self.shard_config.enable_fused_normalization else RMSNorm

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        elif self.tie_weight:
            embedding_cls = PaddingEmbedding

        # the number of heads held by each rank, used by the attention forward for sequence parallelism
        num_q_heads = self.model.config.num_attention_heads
        num_kv_heads = self.model.config.num_key_value_heads
        if sp_mode == "all_to_all":
            num_q_heads //= sp_size
            num_kv_heads //= sp_size
        decoder_attribute_replacement = {}
        if self.shard_config.enable_tensor_parallelism:
            assert (
                self.model.config.num_attention_heads % tp_size == 0
            ), f"The number of attention heads must be divisible by tensor parallel size."
            assert (
                self.model.config.num_key_value_heads % tp_size == 0
            ), f"The number of key_value heads must be divisible by tensor parallel size."
            num_q_heads //= tp_size
            num_kv_heads //= tp_size
            decoder_attribute_replacement["self_attn.hidden_size"] = self.model.config.hidden_size // tp_size
        if sp_mode == "all_to_all" or self.shard_config.enable_tensor_parallelism:
            decoder_attribute_replacement["self_attn.num_heads"] = num_q_heads
            decoder_attribute_replacement["self_attn.num_key_value_heads"] = num_kv_heads
            policy[Qwen3MoeDecoderLayer] = ModulePolicyDescription(attribute_replacement=decoder_attribute_replacement)

        if self.shard_config.enable_tensor_parallelism:
            # tensor parallelism for the attention, the router and the dense mlp layers,
            # the experts of the sparse layers are sharded by EPQwen3MoeSparseMoeBlock
            fp8_communication = self.shard_config.fp8_communication
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=Linear1D_Col,
                        kwargs={"fp8_communication": fp8_communication},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=Linear1D_Col,
                        kwargs={"fp8_communication": fp8_communication},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=Linear1D_Col,
                        kwargs={"fp8_communication": fp8_communication},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                        kwargs={"fp8_communication": fp8_communication},
                    ),
                    # sparse layers
                    SubModuleReplacementDescription(
                        suffix="mlp.gate",
                        target_module=Linear1D_Col,
                        kwargs={"gather_output": True, "fp8_communication": fp8_communication},
                        ignore_if_not_exist=True,
                    ),
                    # dense layers (`mlp_only_layers` or not on the `decoder_sparse_step`)
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=Linear1D_Col,
                        kwargs={"fp8_communication": fp8_communication},
                        ignore_if_not_exist=True,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=Linear1D_Col,
                        kwargs={"fp8_communication": fp8_communication},
                        ignore_if_not_exist=True,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=Linear1D_Row,
                        kwargs={"fp8_communication": fp8_communication},
                        ignore_if_not_exist=True,
                    ),
                ],
                policy=policy,
                target_key=Qwen3MoeDecoderLayer,
            )

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=embedding_cls,
                    kwargs=(
                        {
                            "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by,
                            "fp8_communication": self.shard_config.fp8_communication,
                        }
                        if self.shard_config.enable_tensor_parallelism
                        else {"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by}
                    ),
                ),
                policy=policy,
                target_key=Qwen3MoeModel,
            )

        if self.shard_config.ep_group:
            # expert parallel, dense layers are kept as they are
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="mlp",
                        target_module=EPQwen3MoeSparseMoeBlock,
                        kwargs={
                            "ep_group": self.shard_config.ep_group,
                            "tp_group": self.shard_config.tensor_parallel_process_group,
                            "moe_dp_group": self.shard_config.moe_dp_group,
                            "fp8_communication": self.shard_config.fp8_communication,
                        },
                    )
                ],
                policy=policy,
                target_key=Qwen3MoeDecoderLayer,
            )

        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(suffix="input_layernorm", target_module=norm_cls),
                SubModuleReplacementDescription(suffix="post_attention_layernorm", target_module=norm_cls),
            ],
            policy=policy,
            target_key=Qwen3MoeDecoderLayer,
        )
        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(suffix="norm", target_module=norm_cls),
            policy=policy,
            target_key=Qwen3MoeModel,
        )

        if self.shard_config.enable_flash_attention or self.shard_config.enable_sequence_parallelism:
            # Qwen3MoeAttention is the same as Qwen3Attention
            self.append_or_create_method_replacement(
                description={
                    "forward": get_qwen3_flash_attention_forward(self.shard_config, sp_mode, sp_size, sp_group),
                },
                policy=policy,
                target_key=Qwen3MoeAttention,
            )
            if self.pipeline_stage_manager is None:
                # the model forward prepares the attention mask and splits / gathers the sequence
                self.append_or_create_method_replacement(
                    description={
                        "forward": partial(
                            Qwen3MoePipelineForwards.qwen3_moe_model_forward, shard_config=self.shard_config
                        ),
                    },
                    policy=policy,
                    target_key=Qwen3MoeModel,
                )

        return policy

    def postprocess(self):
        return self.model

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager is None:
            return

        stage_manager = self.pipeline_stage_manager
        if self.model.__class__.__name__ == "Qwen3MoeModel":
            module = self.model
        else:
            module = self.model.model

        layers_per_stage = stage_manager.distribute_layers(len(module.layers))
        if stage_manager.is_interleave:
            # stage_index is passed in by the interleaved schedule for each model chunk
            stage_manager.stage_indices = stage_manager.get_stage_index(layers_per_stage)
            method_replacement = {
                "forward": partial(new_forward, stage_manager=stage_manager, shard_config=self.shard_config)
            }
        else:
            stage_index = stage_manager.get_stage_index(layers_per_stage)
            method_replacement = {
                "forward": partial(
                    new_forward, stage_manager=stage_manager, stage_index=stage_index, shard_config=self.shard_config
                )
            }
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=model_cls)

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "Qwen3MoeModel":
            module = self.model
        else:
            module = self.model.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        held_layers.append(module.rotary_emb)
        layers_per_stage = stage_manager.distribute_layers(len(module.layers))
        if stage_manager.is_interleave:
            assert stage_manager.num_model_chunks is not None
            stage_indices = stage_manager.get_stage_index(layers_per_stage)
            stage_manager.stage_indices = stage_indices
            if stage_manager.is_first_stage(ignore_chunk=True):
                held_layers.append(module.embed_tokens)
            for start_idx, end_idx in stage_indices:
                held_layers.extend(module.layers[start_idx:end_idx])
            if stage_manager.is_last_stage(ignore_chunk=True):
                held_layers.append(module.norm)
        else:
            if stage_manager.is_first_stage():
                held_layers.append(module.embed_tokens)
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
            held_layers.extend(module.layers[start_idx:end_idx])
            if stage_manager.is_last_stage():
                held_layers.append(module.norm)
        return held_layers


class Qwen3MoeModelPolicy(Qwen3MoePolicy):
    def module_policy(self):
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel

        policy = super().module_policy()
        self.set_pipeline_forward(
            model_cls=Qwen3MoeModel,
            new_forward=Qwen3MoePipelineForwards.qwen3_moe_model_forward,
            policy=policy,
        )
        return policy

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in Qwen3-MoE model"""
        return []


class Qwen3MoeForCausalLMPolicy(Qwen3MoePolicy):
    def module_policy(self):
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

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
        self.set_pipeline_forward(
            model_cls=Qwen3MoeForCausalLM,
            new_forward=Qwen3MoePipelineForwards.qwen3_moe_for_causal_lm_forward,
            policy=policy,
        )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        qwen3_moe_model = self.model.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if id(qwen3_moe_model.embed_tokens.weight) == id(self.model.lm_head.weight):
                # tie weights
                return [
                    {
                        0: qwen3_moe_model.embed_tokens.weight,
                        self.pipeline_stage_manager.num_stages - 1: self.model.lm_head.weight,
                    }
                ]
        return []
