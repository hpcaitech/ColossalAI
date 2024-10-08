import warnings
from functools import partial
from typing import Callable, Dict, List, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralModel

from colossalai.shardformer.layer import FusedRMSNorm, Linear1D_Col
from colossalai.shardformer.layer.embedding import PaddingEmbedding, VocabParallelEmbedding1D
from colossalai.shardformer.layer.linear import Linear1D_Row
from colossalai.shardformer.modeling.mixtral import (
    EPMixtralSparseMoeBlock,
    MixtralPipelineForwards,
    get_mixtral_flash_attention_forward,
    get_mixtral_flash_attention_model_forward,
)
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["MixtralPolicy", "MixtralForCausalLMPolicy"]


class MixtralPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.tie_weight = self.tie_weight_check()
        self.origin_attn_implement = self.model.config._attn_implementation
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from transformers.models.mixtral.modeling_mixtral import (
            MixtralAttention,
            MixtralDecoderLayer,
            MixtralFlashAttention2,
            MixtralModel,
            MixtralSdpaAttention,
        )

        ATTN_IMPLEMENTATION = {
            "eager": MixtralAttention,
            "flash_attention_2": MixtralFlashAttention2,
            "sdpa": MixtralSdpaAttention,
        }
        policy = {}
        attn_cls = ATTN_IMPLEMENTATION[self.origin_attn_implement]

        sp_mode = self.shard_config.sequence_parallelism_mode or None
        sp_size = self.shard_config.sequence_parallel_size or None
        sp_group = self.shard_config.sequence_parallel_process_group or None
        sp_partial_derived = sp_mode in ["split_gather", "ring"]
        tp_size = self.shard_config.tensor_parallel_size

        # modified for both SP and TP
        num_q_heads = self.model.config.num_attention_heads
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", None)

        if sp_mode == "all_to_all":
            num_q_heads //= sp_size
            decoder_attribute_replacement = {
                "num_heads": num_q_heads,
            }
            if getattr(self.model.config, "num_key_value_heads", False):
                num_kv_heads //= sp_size
                decoder_attribute_replacement["num_key_value_heads"] = num_kv_heads

            policy[attn_cls] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
            )
        if self.shard_config.enable_sequence_parallelism:
            if self.pipeline_stage_manager is not None:
                # NOTE: we are replacing model forward for both sequence parallelism and pipeline parallelism
                # if both are enabled, one of them will be ignored
                raise NotImplementedError("Sequence parallelism is not supported with pipeline parallelism.")
            self.append_or_create_method_replacement(
                description={
                    "forward": get_mixtral_flash_attention_forward(self.shard_config, sp_mode, sp_size, sp_group),
                },
                policy=policy,
                target_key=attn_cls,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_mixtral_flash_attention_model_forward(
                        self.shard_config,
                        sp_mode=sp_mode,
                        sp_size=sp_size,
                        sp_group=sp_group,
                    ),
                },
                policy=policy,
                target_key=MixtralModel,
            )

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = PaddingEmbedding

        if self.shard_config.enable_tensor_parallelism:
            # tensor parallelism for non-moe params
            assert (
                self.model.config.num_attention_heads % self.shard_config.tensor_parallel_size == 0
            ), f"The number of attention heads must be divisible by tensor parallel size."
            assert (
                self.model.config.num_key_value_heads % self.shard_config.tensor_parallel_size == 0
            ), f"The number of key_value heads must be divisible by tensor parallel size."
            num_q_heads //= tp_size
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": num_q_heads,
            }
            if num_kv_heads:
                num_kv_heads //= tp_size
                decoder_attribute_replacement["self_attn.num_key_value_heads"] = num_kv_heads

            policy[MixtralDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=Linear1D_Col,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=Linear1D_Col,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=Linear1D_Col,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication},
                    ),
                    SubModuleReplacementDescription(
                        suffix="block_sparse_moe.gate",
                        target_module=Linear1D_Col,
                        kwargs={"gather_output": True, "fp8_communication": self.shard_config.fp8_communication},
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
                            "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by,
                            "fp8_communication": self.shard_config.fp8_communication,
                        }
                        if self.shard_config.enable_tensor_parallelism
                        else {"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by}
                    ),
                ),
                policy=policy,
                target_key=MixtralModel,
            )

        if self.shard_config.ep_group:
            # expert parallel
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="block_sparse_moe",
                        target_module=EPMixtralSparseMoeBlock,
                        kwargs={
                            "ep_group": self.shard_config.ep_group,
                            "tp_group": self.shard_config.tensor_parallel_process_group,
                            "moe_dp_group": self.shard_config.moe_dp_group,
                            "fp8_communication": self.shard_config.fp8_communication,
                        },
                    )
                ],
                policy=policy,
                target_key=MixtralDecoderLayer,
            )

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="input_layernorm",
                        target_module=FusedRMSNorm,
                        kwargs={"sp_partial_derived": sp_partial_derived},
                    ),
                    SubModuleReplacementDescription(
                        suffix="post_attention_layernorm",
                        target_module=FusedRMSNorm,
                        kwargs={"sp_partial_derived": sp_partial_derived},
                    ),
                ],
                policy=policy,
                target_key=MixtralDecoderLayer,
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=FusedRMSNorm,
                    kwargs={"sp_partial_derived": sp_partial_derived},
                ),
                policy=policy,
                target_key=MixtralModel,
            )

        if self.shard_config.enable_flash_attention:
            warnings.warn("Flash attention is natively supported in transformers, will ignore the flag.")
            self.shard_config.enable_flash_attention = False

        return policy

    def postprocess(self):
        return self.model

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager:
            if self.shard_config.enable_sequence_parallelism:
                # NOTE: we are replacing model forward for both sequence parallelism and pipeline parallelism
                # if both are enabled, one of them will be ignored
                raise NotImplementedError("Pipeline parallelism is not supported with sequence parallelism.")
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == "MixtralModel":
                module = self.model
            else:
                module = self.model.model

            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
            stage_index = stage_manager.get_stage_index(layers_per_stage)
            method_replacement = {"forward": partial(new_forward, stage_manager=stage_manager, stage_index=stage_index)}
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=model_cls
            )

        return

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "MixtralModel":
            module = self.model
        else:
            module = self.model.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.layers))
        if stage_manager.is_first_stage():
            held_layers.append(module.embed_tokens)
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.norm)

        return held_layers


class MixtralModelPolicy(MixtralPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=MixtralModel,
                new_forward=MixtralPipelineForwards.mixtral_model_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        held_layers = super().get_held_layers()
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in mixtral model"""
        return []


class MixtralForCausalLMPolicy(MixtralPolicy):
    def module_policy(self):
        policy = super().module_policy()
        # TODO: assign pg mesh from plugin to all modules
        if self.shard_config.enable_tensor_parallelism:
            # add a new item for causal lm
            new_item = {
                MixtralForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head",
                            target_module=Linear1D_Col,
                            kwargs=dict(gather_output=True, fp8_communication=self.shard_config.fp8_communication),
                        )
                    ]
                )
            }
            policy.update(new_item)

        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=MixtralForCausalLM,
                new_forward=MixtralPipelineForwards.mixtral_for_causal_lm_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage():
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        mixtral_model = self.model.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if (
                id(mixtral_model.embed_tokens.weight) == id(self.model.lm_head.weight)
                and self.pipeline_stage_manager.num_stages > 1
            ):
                # tie weights
                return [
                    {
                        0: mixtral_model.embed_tokens.weight,
                        self.pipeline_stage_manager.num_stages - 1: self.model.lm_head.weight,
                    }
                ]
        return []


class MixtralForSequenceClassificationPolicy(MixtralPolicy):
    def module_policy(self):
        from transformers import MixtralForSequenceClassification

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for sequence classification
            new_item = {
                MixtralForSequenceClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="score",
                            target_module=Linear1D_Col,
                            kwargs=dict(gather_output=True, fp8_communication=self.shard_config.fp8_communication),
                        )
                    ]
                )
            }
            policy.update(new_item)

        if self.pipeline_stage_manager:
            raise NotImplementedError

        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.score)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in mixtral for sequence classification model"""
        return []
