from functools import partial
from typing import Callable, Dict, List, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from colossalai.shardformer.layer import (
    FusedLayerNorm,
    LayerNorm,
    Linear1D_Col,
    Linear1D_Row,
    LinearWithGradAccum,
    PaddingEmbedding,
    PaddingLMHead,
    VocabParallelEmbedding1D,
    VocabParallelLMHead1D,
)

from ..modeling.command import (
    CommandPipelineForwards,
    get_command_flash_attention_forward,
    get_command_flash_attention_model_forward,
    get_lm_forward_with_dist_cross_entropy,
)
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["CommandPolicy", "CommandForCausalLMPolicy"]


class CommandPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.tie_weight = self.tie_weight_check()
        self.origin_attn_implement = self.model.config._attn_implementation
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from transformers.models.cohere.modeling_cohere import (
            CohereAttention,
            CohereDecoderLayer,
            CohereFlashAttention2,
            CohereModel,
            CohereSdpaAttention,
        )

        ATTN_IMPLEMENTATION = {
            "eager": CohereAttention,
            "flash_attention_2": CohereFlashAttention2,
            "sdpa": CohereSdpaAttention,
        }
        policy = {}

        attn_cls = ATTN_IMPLEMENTATION[self.origin_attn_implement]
        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = PaddingEmbedding

        if self.shard_config.enable_fused_normalization:
            norm_cls = FusedLayerNorm
        else:
            norm_cls = LayerNorm

        sp_mode = self.shard_config.sequence_parallelism_mode or None
        sp_size = self.shard_config.sequence_parallel_size or None
        sp_group = self.shard_config.sequence_parallel_process_group or None
        sp_partial_derived = sp_mode in ["split_gather", "ring"]
        if sp_mode == "ring_attn" and not self.is_causal:
            raise ValueError("Ring attention is only meant for causal language modeling.")

        tp_size = self.shard_config.tensor_parallel_size or None
        num_q_heads = self.model.config.num_attention_heads
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", None)
        if sp_mode == "all_to_all":
            num_q_heads //= sp_size
            decoder_attribute_replacement = {"num_heads": num_q_heads}
            if num_kv_heads:
                num_kv_heads //= sp_size
                decoder_attribute_replacement["num_key_value_heads"] = num_kv_heads

            policy[attn_cls] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
            )
        if self.shard_config.enable_flash_attention or self.shard_config.enable_sequence_parallelism:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_command_flash_attention_forward(self.shard_config, sp_mode, sp_size, sp_group),
                },
                policy=policy,
                target_key=attn_cls,
            )
            if self.pipeline_stage_manager is None:
                self.append_or_create_method_replacement(
                    description={
                        "forward": get_command_flash_attention_model_forward(
                            self.shard_config,
                            sp_mode=sp_mode,
                            sp_size=sp_size,
                            sp_group=sp_group,
                        ),
                    },
                    policy=policy,
                    target_key=CohereModel,
                )

        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv

        if self.shard_config.enable_tensor_parallelism:
            assert (
                num_q_heads % tp_size == 0
            ), f"The number of attention heads must be divisible by tensor parallel size."
            if hasattr(self.model.config, "num_key_value_heads"):
                assert (
                    num_kv_heads >= tp_size and num_kv_heads % tp_size == 0
                ), f"The number of key_value heads must be divisible by, and must not be less than tensor parallel size."
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size // tp_size,
                "self_attn.num_heads": num_q_heads // tp_size,
            }
            if getattr(self.model.config, "num_key_value_heads", False):
                decoder_attribute_replacement["self_attn.num_key_value_heads"] = num_kv_heads // tp_size

            policy[CohereDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=Linear1D_Row,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
                    ),
                ],
            )
        elif use_zbv:
            policy[CohereDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=LinearWithGradAccum,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=LinearWithGradAccum,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=LinearWithGradAccum,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=LinearWithGradAccum,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=LinearWithGradAccum,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=LinearWithGradAccum,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=LinearWithGradAccum,
                        kwargs=dict(
                            seq_parallel_mode=sp_mode,
                            fp8_communication=self.shard_config.fp8_communication,
                            use_zbv=use_zbv,
                        ),
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
                target_key=CohereModel,
            )

        # optimization configuration
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="input_layernorm",
                    target_module=norm_cls,
                    kwargs={"sp_partial_derived": sp_partial_derived},
                ),
            ],
            policy=policy,
            target_key=CohereDecoderLayer,
        )

        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(
                suffix="norm",
                target_module=norm_cls,
                kwargs={"sp_partial_derived": sp_partial_derived},
            ),
            policy=policy,
            target_key=CohereModel,
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
        if self.model.__class__.__name__ == "CohereModel":
            module = self.model
        else:
            module = self.model.model

        if stage_manager.is_interleave:
            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
            stage_manager.stage_indices = stage_manager.get_stage_index(layers_per_stage)
            method_replacement = {
                "forward": partial(new_forward, stage_manager=stage_manager, shard_config=self.shard_config)
            }

        else:
            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
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

        if self.model.__class__.__name__ == "CohereModel":
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


class CommandModelPolicy(CommandPolicy):
    def module_policy(self):
        policy = super().module_policy()
        from transformers.models.cohere.modeling_cohere import CohereModel

        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=CohereModel, new_forward=CommandPipelineForwards.command_model_forward, policy=policy
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        held_layers = super().get_held_layers()
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in command model"""
        return []


class CommandForCausalLMPolicy(CommandPolicy):
    def module_policy(self):
        from transformers import CohereForCausalLM

        self.is_causal = True
        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for causal lm
            new_item = {
                CohereForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head",
                            target_module=VocabParallelLMHead1D,
                            kwargs={
                                "gather_output": not self.shard_config.parallel_output,
                                "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by,
                                "fp8_communication": self.shard_config.fp8_communication,
                            },
                        )
                    ],
                )
            }
            if self.shard_config.parallel_output:
                new_item[CohereForCausalLM].method_replacement = {
                    "forward": get_lm_forward_with_dist_cross_entropy(self.shard_config)
                }
        else:
            new_item = {
                CohereForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head",
                            target_module=PaddingLMHead,
                            kwargs={"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by},
                        )
                    ],
                )
            }
        policy.update(new_item)

        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=CohereForCausalLM,
                new_forward=CommandPipelineForwards.command_for_causal_lm_forward,
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
            if stage_manager.is_last_stage(ignore_chunk=True):
                held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        command_model = self.model.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if (
                id(command_model.embed_tokens.weight) == id(self.model.lm_head.weight)
                and self.pipeline_stage_manager.num_stages > 1
            ):
                # tie weights
                return [
                    {
                        0: command_model.embed_tokens.weight,
                        self.pipeline_stage_manager.num_stages - 1: self.model.lm_head.weight,
                    }
                ]
        return []
