from functools import partial
from typing import Callable, Dict, List

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

import colossalai.shardformer.layer as col_nn

from ..modeling.bloom import (
    BloomPipelineForwards,
    build_bloom_alibi_tensor_fn,
    get_bloom_flash_attention_forward,
    get_bloom_sequence_parallel_forward_fn,
    get_jit_fused_bloom_attention_forward,
    get_jit_fused_bloom_gelu_forward,
    get_jit_fused_bloom_mlp_forward,
)
from ..modeling.jit import get_dropout_add_func, get_jit_fused_dropout_add_func, get_jit_fused_gelu_forward_func
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription


class BloomPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()
        import transformers
        from packaging.version import Version

        assert Version(transformers.__version__) <= Version(
            "4.33.0"
        ), "The Bloom model should run on a transformers version not greater than 4.33.0."

    def config_sanity_check(self):
        pass

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        if self.shard_config.enable_tensor_parallelism:
            vocab_size = self.model.config.vocab_size
            world_size = self.shard_config.tensor_parallel_size
            if vocab_size % world_size != 0:
                new_vocab_size = vocab_size + world_size - vocab_size % world_size
                self.model.resize_token_embeddings(new_vocab_size)
        return self.model

    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomGelu, BloomMLP, BloomModel

        policy = {}

        if self.shard_config.enable_fused_normalization:
            norm_cls = col_nn.FusedLayerNorm
        else:
            norm_cls = col_nn.LayerNorm

        sp_mode = self.shard_config.sequence_parallelism_mode if self.shard_config.enable_sequence_parallelism else None
        overlap = self.shard_config.enable_sequence_overlap
        sp_partial_derived = sp_mode in ["1"]

        if self.shard_config.enable_tensor_parallelism:
            policy[BloomBlock] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attention.hidden_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "self_attention.split_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "self_attention.num_heads": self.model.config.n_head // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attention.query_key_value",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={"seq_parallel_mode": sp_mode, "overlap": overlap},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.dense",
                        target_module=col_nn.Linear1D_Row,
                        kwargs={"seq_parallel_mode": sp_mode},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.attention_dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dense_h_to_4h",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={"seq_parallel_mode": sp_mode, "overlap": overlap},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dense_4h_to_h",
                        target_module=col_nn.Linear1D_Row,
                        kwargs={"seq_parallel_mode": sp_mode},
                    ),
                ],
            )

            policy[BloomModel] = ModulePolicyDescription(
                attribute_replacement={
                    "num_heads": self.model.config.n_head // self.shard_config.tensor_parallel_size,
                },
                method_replacement={
                    "build_alibi_tensor": build_bloom_alibi_tensor_fn(self.shard_config.tensor_parallel_process_group)
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="word_embeddings",
                        target_module=col_nn.VocabParallelEmbedding1D,
                    )
                ],
            )

        # optimization configuration
        # handle bloom model
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="ln_f",
                    target_module=norm_cls,
                ),
                SubModuleReplacementDescription(
                    suffix="word_embeddings_layernorm",
                    target_module=norm_cls,
                ),
            ],
            policy=policy,
            target_key=BloomModel,
        )

        # handle bloom block
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
            target_key=BloomBlock,
        )

        if sp_mode == "1":
            self.append_or_create_method_replacement(
                description={"forward": get_bloom_sequence_parallel_forward_fn(self.shard_config)},
                policy=policy,
                target_key=BloomModel,
            )

        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_bloom_flash_attention_forward(),
                    "dropout_add": get_dropout_add_func(),
                },
                policy=policy,
                target_key=BloomAttention,
            )

        # enable jit fused operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_bloom_attention_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=BloomAttention,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_bloom_mlp_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=BloomMLP,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_bloom_gelu_forward(),
                    "bloom_gelu_forward": get_jit_fused_gelu_forward_func(),
                },
                policy=policy,
                target_key=BloomGelu,
            )

        return policy

    def postprocess(self):
        return self.model

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == "BloomModel":
                module = self.model
            else:
                module = self.model.transformer

            layers_per_stage = Policy.distribute_layers(len(module.h), stage_manager.num_stages)
            stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
            method_replacement = {
                "forward": partial(
                    new_forward, stage_manager=stage_manager, stage_index=stage_index, shard_config=self.shard_config
                )
            }
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=model_cls
            )
        return

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "BloomModel":
            module = self.model
        else:
            module = self.model.transformer
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.h), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.word_embeddings)
            held_layers.append(module.word_embeddings_layernorm)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.h[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.ln_f)

        return held_layers


class BloomModelPolicy(BloomPolicy):
    def module_policy(self):
        policy = super().module_policy()
        from transformers.models.bloom.modeling_bloom import BloomModel

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BloomModel, new_forward=BloomPipelineForwards.bloom_model_forward, policy=policy
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """no shared params in bloom model"""
        return []


class BloomForCausalLMPolicy(BloomPolicy):
    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomForCausalLM

        policy = super().module_policy()

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="lm_head", target_module=col_nn.Linear1D_Col, kwargs=dict(gather_output=True)
                ),
                policy=policy,
                target_key=BloomForCausalLM,
            )
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BloomForCausalLM, new_forward=BloomPipelineForwards.bloom_for_causal_lm_forward, policy=policy
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
        bloom_model = self.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if id(bloom_model.transformer.word_embeddings.weight) == id(bloom_model.lm_head.weight):
                # tie weights
                return [
                    {
                        0: bloom_model.transformer.word_embeddings.weight,
                        self.pipeline_stage_manager.num_stages - 1: bloom_model.lm_head.weight,
                    }
                ]
        return []


class BloomForSequenceClassificationPolicy(BloomPolicy):
    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomForSequenceClassification

        policy = super().module_policy()

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="score", target_module=col_nn.Linear1D_Col, kwargs=dict(gather_output=True)
                ),
                policy=policy,
                target_key=BloomForSequenceClassification,
            )
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BloomForSequenceClassification,
                new_forward=BloomPipelineForwards.bloom_for_sequence_classification_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage():
            held_layers.append(self.model.score)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in bloom for sequence classification model"""
        return []


class BloomForTokenClassificationPolicy(BloomPolicy):
    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomForTokenClassification

        policy = super().module_policy()

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="classifier", target_module=col_nn.Linear1D_Col, kwargs=dict(gather_output=True)
                    ),
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    ),
                ],
                policy=policy,
                target_key=BloomForTokenClassification,
            )
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BloomForTokenClassification,
                new_forward=BloomPipelineForwards.bloom_for_token_classification_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage():
            held_layers.append(self.model.dropout)
            held_layers.append(self.model.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in bloom for token classification model"""
        return []


class BloomForQuestionAnsweringPolicy(BloomPolicy):
    # No head sharding as the output features is only 2
    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomForQuestionAnswering

        policy = super().module_policy()
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BloomForQuestionAnswering,
                new_forward=BloomPipelineForwards.bloom_for_question_answering_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage():
            held_layers.append(self.model.qa_outputs)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in bloom for question answering model"""
        return []
