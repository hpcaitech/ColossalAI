from functools import partial
from typing import Callable, Dict, List, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from transformers.utils import is_flash_attn_greater_or_equal_2_10

from colossalai.shardformer.layer import FusedRMSNorm, Linear1D_Col, LinearWithGradAccum
from colossalai.shardformer.layer.embedding import PaddingEmbedding, VocabParallelEmbedding1D
from colossalai.shardformer.layer.linear import Linear1D_Row
from colossalai.shardformer.modeling.deepseek import (
    DeepseekMoEGate_Col,
    DeepseekPipelineForwards,
    EPDeepseekMoE,
    get_deepseek_flash_attention_forward,
    get_deepseek_flash_attention_model_forward,
)
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["DeepseekPolicy", "DeepseekForCausalLMPolicy"]


class DeepseekPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.tie_weight = self.tie_weight_check()
        self.origin_attn_implement = self.model.config._attn_implementation
        """
        Because transformers library's bug for AutoModel/AutoConfig, who pop “attn_implement” twice from modeling_utils.py and configuration_utils.py.
        This bug causes attn_cls to be set to sdpa. Here we assign it to "flash_attention_2".
        """
        # self.origin_attn_implement =  "flash_attention_2"
        if self.shard_config.enable_tensor_parallelism:
            # Resize embedding
            vocab_size = self.model.config.vocab_size
            world_size = self.shard_config.tensor_parallel_size

            if vocab_size % world_size != 0:
                new_vocab_size = vocab_size + world_size - vocab_size % world_size
                self.model.resize_token_embeddings(new_vocab_size)

        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:

        ATTN_IMPLEMENTATION = {
            "eager": "DeepseekAttention",
            "flash_attention_2": "DeepseekFlashAttention2",
            "sdpa": "DeepseekSdpaAttention",
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
                    "forward": get_deepseek_flash_attention_forward(self.shard_config, sp_mode, sp_size, sp_group),
                },
                policy=policy,
                target_key=attn_cls,
            )
            if self.pipeline_stage_manager is None:
                self.append_or_create_method_replacement(
                    description={
                        "forward": get_deepseek_flash_attention_model_forward(
                            self.shard_config,
                            sp_mode=sp_mode,
                            sp_size=sp_size,
                            sp_group=sp_group,
                        ),
                    },
                    policy=policy,
                    target_key="DeepseekModel",
                )
        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = PaddingEmbedding

        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv

        if self.shard_config.enable_tensor_parallelism:
            # tensor parallelism for non-moe params
            assert (
                self.model.config.num_attention_heads % self.shard_config.tensor_parallel_size == 0
            ), f"The number of attention heads must be divisible by tensor parallel size."
            assert (
                self.model.config.num_key_value_heads % self.shard_config.tensor_parallel_size == 0
            ), f"The number of key_value heads must be divisible by tensor parallel size."
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
            }
            num_q_heads //= tp_size
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": num_q_heads,
            }
            if num_kv_heads:
                num_kv_heads //= tp_size
                decoder_attribute_replacement["self_attn.num_key_value_heads"] = num_kv_heads

            policy["DeepseekDecoderLayer"] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=Linear1D_Col,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication, "use_zbv": use_zbv},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=Linear1D_Col,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication, "use_zbv": use_zbv},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=Linear1D_Col,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication, "use_zbv": use_zbv},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication, "use_zbv": use_zbv},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate",
                        target_module=DeepseekMoEGate_Col,
                        kwargs={
                            "gather_output": True,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "config": self.model.config,
                        },
                        ignore_if_not_exist=True,
                    ),
                ],
            )
        elif use_zbv:
            policy["DeepseekDecoderLayer"] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=LinearWithGradAccum,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication, "use_zbv": use_zbv},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=LinearWithGradAccum,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication, "use_zbv": use_zbv},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=LinearWithGradAccum,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication, "use_zbv": use_zbv},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=LinearWithGradAccum,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication, "use_zbv": use_zbv},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate",
                        target_module=DeepseekMoEGate_Col,
                        kwargs={
                            "gather_output": True,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "config": self.model.config,
                        },
                        ignore_if_not_exist=True,
                    ),
                ],
            )
        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=embedding_cls,
                    kwargs={
                        "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by,
                        "fp8_communication": self.shard_config.fp8_communication,
                    },
                ),
                policy=policy,
                target_key="DeepseekModel",
            )

        if self.shard_config.ep_group:
            # expert parallel
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="mlp",
                        target_module=EPDeepseekMoE,
                        kwargs={
                            "ep_group": self.shard_config.ep_group,
                            "tp_group": self.shard_config.tensor_parallel_process_group,
                            "moe_dp_group": self.shard_config.moe_dp_group,
                            "fp8_communication": self.shard_config.fp8_communication,
                        },
                    )
                ],
                policy=policy,
                target_key="DeepseekDecoderLayer",
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
                target_key="DeepseekDecoderLayer",
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=FusedRMSNorm,
                    kwargs={"sp_partial_derived": sp_partial_derived},
                ),
                policy=policy,
                target_key="DeepseekModel",
            )

        if self.shard_config.enable_flash_attention:
            # NOTE: there is a bug for toggling flash attention in AutoModel, which has to be used for deepseek right now
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            flash_attn_cls = get_class_from_dynamic_module(
                "deepseek-ai/deepseek-moe-16b-base--modeling_deepseek.DeepseekFlashAttention2",
                "deepseek-ai/deepseek-moe-16b-base",
            )

            class TargetFlashAttn:
                def __init__(self):
                    raise RuntimeError("This class should not be instantiated")

                @staticmethod
                def from_native_module(original_attn: nn.Module, *args, **kwargs) -> nn.Module:
                    original_attn.__class__ = flash_attn_cls
                    original_attn._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
                    return original_attn

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="self_attn",
                    target_module=TargetFlashAttn,
                ),
                policy=policy,
                target_key="DeepseekDecoderLayer",
            )
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
            if self.model.__class__.__name__ == "DeepseekModel":
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

        if self.model.__class__.__name__ == "DeepseekModel":
            module = self.model
        else:
            module = self.model.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        if stage_manager.is_interleave:
            assert stage_manager.num_model_chunks is not None
            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
            stage_indices = stage_manager.get_stage_index(layers_per_stage)
            if stage_manager.is_first_stage(ignore_chunk=True):
                held_layers.append(module.embed_tokens)
            for start_idx, end_idx in stage_indices:
                held_layers.extend(module.layers[start_idx:end_idx])
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(module.norm)
        else:
            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
            if stage_manager.is_first_stage():
                held_layers.append(module.embed_tokens)
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
            held_layers.extend(module.layers[start_idx:end_idx])
            if stage_manager.is_last_stage():
                held_layers.append(module.norm)

        return held_layers


class DeepseekModelPolicy(DeepseekPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls="DeepseekModel",
                new_forward=DeepseekPipelineForwards.deepseek_model_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        held_layers = super().get_held_layers()
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in llama model"""
        return []


class DeepseekForCausalLMPolicy(DeepseekPolicy):
    def module_policy(self):
        policy = super().module_policy()
        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv
        # TODO: assign pg mesh from plugin to all modules
        if self.shard_config.enable_tensor_parallelism:
            # add a new item for casual lm
            new_item = {
                "DeepseekForCausalLM": ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head",
                            target_module=Linear1D_Col,
                            kwargs=dict(
                                gather_output=True,
                                fp8_communication=self.shard_config.fp8_communication,
                                use_zbv=use_zbv,
                            ),
                        )
                    ]
                )
            }
            policy.update(new_item)
        elif use_zbv:
            # add a new item for casual lm
            new_item = {
                "DeepseekForCausalLM": ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head",
                            target_module=LinearWithGradAccum,
                            kwargs=dict(
                                gather_output=True,
                                fp8_communication=self.shard_config.fp8_communication,
                                use_zbv=use_zbv,
                            ),
                        )
                    ]
                )
            }
            policy.update(new_item)

        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls="DeepseekForCausalLM",
                new_forward=DeepseekPipelineForwards.deepseek_for_causal_lm_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_interleave:
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(self.model.lm_head)
        else:
            if stage_manager.is_last_stage():
                held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        deepseek_model = self.model.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if (
                id(deepseek_model.embed_tokens.weight) == id(self.model.lm_head.weight)
                and self.pipeline_stage_manager.num_stages > 1
            ):
                # tie weights
                return [
                    {
                        0: deepseek_model.embed_tokens.weight,
                        self.pipeline_stage_manager.num_stages - 1: self.model.lm_head.weight,
                    }
                ]
        return []
