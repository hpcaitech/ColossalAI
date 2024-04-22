import warnings
from functools import partial
from typing import Callable, Dict, List, Union

import torch.nn as nn
from torch import Tensor

import colossalai.shardformer.layer as col_nn
from colossalai.shardformer.modeling.chatglm2 import ChatGLMPipelineForwards

from ..modeling.chatglm2 import (
    get_chatglm_sequence_parallel_forward_fn,
    get_flash_core_attention_forward,
    get_jit_fused_glm_block_forward,
)
from ..modeling.jit import get_jit_fused_dropout_add_func
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    "ChatGLMPolicy",
    "ChatGLMModelPolicy",
    "ChatGLMForConditionalGenerationPolicy",
]


class ChatGLMPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        if self.pipeline_stage_manager is not None:
            # the batch_size_dim is bounded to Model
            bsz_dim = 1
            setattr(self.model, "batch_size_dim", bsz_dim)

        self.tie_weight = self.tie_weight_check()
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        policy = {}

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = col_nn.VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = col_nn.PaddingEmbedding

        if self.shard_config.enable_fused_normalization:
            if self.model.config.rmsnorm:
                norm_cls = col_nn.FusedRMSNorm
            else:
                norm_cls = col_nn.FusedLayerNorm
        else:
            if self.model.config.rmsnorm:
                norm_cls = col_nn.RMSNorm
            else:
                norm_cls = col_nn.LayerNorm

        sp_mode = self.shard_config.sequence_parallelism_mode if self.shard_config.enable_sequence_parallelism else None
        assert sp_mode != "all_to_all", "all_to_all sequence parallelism is not supported for ChatGLM2"
        if sp_mode == "ring":
            warnings.warn(
                f"For ChatGLM2, sequence parallelism is currently not support mode {sp_mode}, will set to be split_gather"
            )
            sp_mode = "split_gather"
        overlap = self.shard_config.enable_sequence_overlap
        sp_partial_derived = sp_mode == "split_gather"

        if self.shard_config.enable_tensor_parallelism:
            policy["ChatGLMModel"] = ModulePolicyDescription(
                attribute_replacement={},
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="embedding.word_embeddings",
                        target_module=col_nn.VocabParallelEmbedding1D,
                    )
                ],
            )
            assert (
                self.model.config.num_attention_heads % self.shard_config.tensor_parallel_size == 0
            ), f"num_attention_heads {self.model.config.num_attention_heads} should be divisible by tensor_parallel_size {self.shard_config.tensor_parallel_size}"
            attn_kwargs = {
                "self_attention.qkv_hidden_size": (
                    self.model.config.kv_channels * self.model.config.num_attention_heads * 3
                )
                // self.shard_config.tensor_parallel_size,
            }
            if self.model.config.multi_query_attention:
                assert (
                    self.model.config.multi_query_group_num % self.shard_config.tensor_parallel_size == 0
                ), f"multi_query_group_num {self.model.config.multi_query_group_num} should be divisible by tensor_parallel_size {self.shard_config.tensor_parallel_size}"
                attn_kwargs["self_attention.num_multi_query_groups_per_partition"] = (
                    self.model.config.multi_query_group_num // self.shard_config.tensor_parallel_size
                )
                attn_kwargs["self_attention.qkv_hidden_size"] = (
                    self.model.config.kv_channels * self.model.config.num_attention_heads
                    + 2 * self.model.config.kv_channels * self.model.config.multi_query_group_num
                ) // self.shard_config.tensor_parallel_size
            policy["GLMBlock"] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attention.num_attention_heads_per_partition": self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "self_attention.projection_size": (
                        self.model.config.kv_channels * self.model.config.num_attention_heads
                    )
                    // self.shard_config.tensor_parallel_size,
                    "self_attention.core_attention.num_attention_heads_per_partition": self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "self_attention.core_attention.hidden_size_per_partition": self.model.config.kv_channels
                    * self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    **attn_kwargs,
                },
                param_replacement=[],
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attention.query_key_value",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "seq_parallel_dim": 0,
                            "overlap": overlap,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.dense",
                        target_module=col_nn.Linear1D_Row,
                        kwargs={"seq_parallel_mode": sp_mode, "seq_parallel_dim": 0},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.core_attention.attention_dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                ],
            )

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="embedding.word_embeddings",
                        target_module=embedding_cls,
                        kwargs={"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by},
                    ),
                ],
                policy=policy,
                target_key=ChatGLMModel,
            )
        # optimization configuration
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="input_layernorm",
                    target_module=norm_cls,
                    kwargs={"sp_partial_derived": sp_partial_derived},
                ),
                SubModuleReplacementDescription(
                    suffix="post_attention_layernorm",
                    target_module=norm_cls,
                    kwargs={"sp_partial_derived": sp_partial_derived},
                ),
            ],
            policy=policy,
            target_key="GLMBlock",
        )

        if self.model.config.post_layer_norm:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="encoder.final_layernorm",
                        target_module=norm_cls,
                    )
                ],
                policy=policy,
                target_key="ChatGLMModel",
            )

        # use flash attention
        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_flash_core_attention_forward(),
                },
                policy=policy,
                target_key="CoreAttention",
            )

        # use sequence parallel
        if sp_mode == "split_gather":
            self.append_or_create_method_replacement(
                description={"forward": get_chatglm_sequence_parallel_forward_fn(self.shard_config)},
                policy=policy,
                target_key="ChatGLMModel",
            )

        # use jit fused operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_glm_block_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key="GLMBlock",
            )

        return policy

    def postprocess(self):
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "ChatGLMModel":
            module = self.model
        else:
            module = self.model.transformer
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(module.num_layers)
        if stage_manager.is_first_stage():
            held_layers.append(module.embedding)
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.encoder.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            if module.encoder.post_layer_norm:
                held_layers.append(module.encoder.final_layernorm)

        # rotary_pos_emb is needed for all stages
        held_layers.append(module.rotary_pos_emb)

        return held_layers

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if not self.pipeline_stage_manager:
            raise ValueError("set_pipeline_forward method can only be called when pipeline parallel is enabled.")
        stage_manager = self.pipeline_stage_manager
        if self.model.__class__.__name__ == "ChatGLMModel":
            module = self.model
        else:
            module = self.model.transformer

        layers_per_stage = stage_manager.distribute_layers(module.num_layers)
        stage_index = stage_manager.get_stage_index(layers_per_stage)
        method_replacement = {
            "forward": partial(
                new_forward,
                stage_manager=stage_manager,
                stage_index=stage_index,
                shard_config=self.shard_config,
            )
        }
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=model_cls)


class ChatGLMModelPolicy(ChatGLMPolicy):
    def module_policy(self):
        pass

        policy = super().module_policy()

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls="ChatGLMModel",
                new_forward=ChatGLMPipelineForwards.chatglm_model_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in ChatGLMModel."""
        return []


class ChatGLMForConditionalGenerationPolicy(ChatGLMModelPolicy):
    def module_policy(self):
        policy = super().module_policy()

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls="ChatGLMForConditionalGeneration",
                new_forward=ChatGLMPipelineForwards.chatglm_for_conditional_generation_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.transformer.output_layer)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in ChatGLMForConditionalGenerationModel."""
        return []
