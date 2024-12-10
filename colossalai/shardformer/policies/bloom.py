import warnings
from functools import partial
from typing import Callable, Dict, List

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

import colossalai.shardformer.layer as col_nn

from ..modeling.bloom import (
    BloomPipelineForwards,
    build_bloom_alibi_tensor_fn,
    get_bloom_sequence_parallel_forward_fn,
    get_jit_fused_bloom_attention_forward,
    get_jit_fused_bloom_gelu_forward,
    get_jit_fused_bloom_mlp_forward,
    get_lm_forward_with_dist_cross_entropy,
)
from ..modeling.jit import get_jit_fused_dropout_add_func, get_jit_fused_gelu_forward_func
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription


class BloomPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()

    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.tie_weight = self.tie_weight_check()
        return self.model

    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomGelu, BloomMLP, BloomModel

        policy = {}

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = col_nn.VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = col_nn.PaddingEmbedding

        if self.shard_config.enable_fused_normalization:
            norm_cls = col_nn.FusedLayerNorm
        else:
            norm_cls = col_nn.LayerNorm

        sp_mode = self.shard_config.sequence_parallelism_mode or None
        assert sp_mode != "all_to_all", "all_to_all sequence parallelism is not supported for BLOOM"
        if sp_mode == "ring":
            warnings.warn(
                f"For BLOOM, sequence parallelism is currently not support mode {sp_mode}, will set to be split_gather"
            )
            sp_mode = "split_gather"

        sp_partial_derived = sp_mode == "split_gather"

        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv

        if self.shard_config.enable_tensor_parallelism:
            assert (
                self.model.config.n_head % self.shard_config.tensor_parallel_size == 0
            ), f"The number of attention heads must be divisible by tensor parallel size."
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
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.dense",
                        target_module=col_nn.Linear1D_Row,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.attention_dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dense_h_to_4h",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dense_4h_to_h",
                        target_module=col_nn.Linear1D_Row,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
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
            )

        if use_zbv:
            policy[BloomBlock] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attention.query_key_value",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.dense",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.attention_dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dense_h_to_4h",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dense_4h_to_h",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                ],
            )

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="word_embeddings",
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
                ],
                policy=policy,
                target_key=BloomModel,
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

        if sp_mode == "split_gather":
            self.append_or_create_method_replacement(
                description={"forward": get_bloom_sequence_parallel_forward_fn(self.shard_config)},
                policy=policy,
                target_key=BloomModel,
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

            layers_per_stage = stage_manager.distribute_layers(len(module.h))
            stage_index = stage_manager.get_stage_index(layers_per_stage)
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
        if stage_manager.is_interleave:
            layers_per_stage = stage_manager.distribute_layers(len(module.h))
            stage_indices = stage_manager.get_stage_index(layers_per_stage)
            if stage_manager.is_first_stage(ignore_chunk=True):
                held_layers.append(module.word_embeddings)
                held_layers.append(module.word_embeddings_layernorm)
            for start_idx, end_idx in stage_indices:
                held_layers.extend(module.h[start_idx:end_idx])
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(module.ln_f)
        else:
            layers_per_stage = stage_manager.distribute_layers(len(module.h))
            if stage_manager.is_first_stage():
                held_layers.append(module.word_embeddings)
                held_layers.append(module.word_embeddings_layernorm)
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
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
                    suffix="lm_head",
                    target_module=col_nn.VocabParallelLMHead1D,
                    kwargs=dict(
                        gather_output=not self.shard_config.parallel_output,
                        make_vocab_size_divisible_by=self.shard_config.make_vocab_size_divisible_by,
                        fp8_communication=self.shard_config.fp8_communication,
                    ),
                ),
                policy=policy,
                target_key=BloomForCausalLM,
            )
            if self.shard_config.parallel_output:
                method_replacement = {"forward": get_lm_forward_with_dist_cross_entropy(self.shard_config)}
                self.append_or_create_method_replacement(
                    description=method_replacement, policy=policy, target_key=BloomForCausalLM
                )
        else:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="lm_head",
                    target_module=col_nn.PaddingLMHead,
                    kwargs=dict(make_vocab_size_divisible_by=self.shard_config.make_vocab_size_divisible_by),
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
        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="score",
                    target_module=col_nn.Linear1D_Col,
                    kwargs=dict(gather_output=True, fp8_communication=self.shard_config.fp8_communication),
                ),
                policy=policy,
                target_key=BloomForSequenceClassification,
            )
        elif use_zbv:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="score",
                    target_module=col_nn.LinearWithGradAccum,
                    kwargs=dict(
                        gather_output=True, fp8_communication=self.shard_config.fp8_communication, use_zbv=use_zbv
                    ),
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
        if stage_manager.is_interleave:
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(self.model.score)
        else:
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
        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="classifier",
                        target_module=col_nn.Linear1D_Col,
                        kwargs=dict(gather_output=True, fp8_communication=self.shard_config.fp8_communication),
                    ),
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    ),
                ],
                policy=policy,
                target_key=BloomForTokenClassification,
            )
        elif use_zbv:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="classifier",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs=dict(
                            gather_output=True, fp8_communication=self.shard_config.fp8_communication, use_zbv=use_zbv
                        ),
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
        if stage_manager.is_interleave:
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(self.model.dropout)
                held_layers.append(self.model.classifier)
        else:
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
        if stage_manager.is_interleave:
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(self.model.qa_outputs)
        else:
            if stage_manager.is_last_stage():
                held_layers.append(self.model.qa_outputs)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in bloom for question answering model"""
        return []
