import warnings
from functools import partial
from typing import Callable, Dict, List

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

import colossalai.shardformer.layer as col_nn

from ..modeling.bert import (
    BertPipelineForwards,
    bert_sequence_parallel_forward_fn,
    get_jit_fused_bert_intermediate_forward,
    get_jit_fused_bert_output_forward,
    get_jit_fused_bert_self_output_forward,
)
from ..modeling.jit import get_jit_fused_dropout_add_func
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    "BertPolicy",
    "BertModelPolicy",
    "BertForPreTrainingPolicy",
    "BertLMHeadModelPolicy",
    "BertForMaskedLMPolicy",
    "BertForNextSentencePredictionPolicy",
    "BertForSequenceClassificationPolicy",
    "BertForTokenClassificationPolicy",
    "BertForMultipleChoicePolicy",
    "BertForQuestionAnsweringPolicy",
]


class BertPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.tie_weight = self.tie_weight_check()
        self.enable_bias_gelu_fused = self.shard_config.enable_jit_fused and self.model.config.hidden_act == "gelu"
        return self.model

    def module_policy(self):
        from transformers.models.bert.modeling_bert import (
            BertEmbeddings,
            BertIntermediate,
            BertLayer,
            BertModel,
            BertOutput,
            BertSelfOutput,
        )

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
        assert sp_mode != "all_to_all", "all_to_all sequence parallelism is not supported for Bert"
        if sp_mode == "ring":
            warnings.warn(
                f"For Bert, sequence parallelism is currently not support mode {sp_mode}, will set to be split_gather"
            )
            sp_mode = "split_gather"

        sp_partial_derived = sp_mode == "split_gather"

        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv

        if self.shard_config.enable_tensor_parallelism:
            assert (
                self.model.config.num_attention_heads % self.shard_config.tensor_parallel_size == 0
            ), f"The number of attention heads must be divisible by tensor parallel size."
            policy[BertLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "attention.self.all_head_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "crossattention.self.all_head_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "attention.self.num_attention_heads": self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "crossattention.self.num_attention_heads": self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attention.self.query",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.self.key",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.self.value",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.self.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dense",
                        target_module=col_nn.Linear1D_Row,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="intermediate.dense",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "skip_bias_add": self.enable_bias_gelu_fused,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="output.dense",
                        target_module=col_nn.Linear1D_Row,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="output.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                ],
            )

            policy[BertEmbeddings] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    ),
                ]
            )
            if self.enable_bias_gelu_fused:
                self.append_or_create_method_replacement(
                    description={
                        "forward": get_jit_fused_bert_intermediate_forward(),
                    },
                    policy=policy,
                    target_key=BertIntermediate,
                )

        elif use_zbv:
            policy[BertLayer] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attention.self.query",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.self.key",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.self.value",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.self.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dense",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="intermediate.dense",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "skip_bias_add": self.enable_bias_gelu_fused,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="output.dense",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="output.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                ],
            )

            policy[BertEmbeddings] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    ),
                ]
            )
            if self.enable_bias_gelu_fused:
                self.append_or_create_method_replacement(
                    description={
                        "forward": get_jit_fused_bert_intermediate_forward(),
                    },
                    policy=policy,
                    target_key=BertIntermediate,
                )

        if sp_mode == "split_gather":
            self.append_or_create_method_replacement(
                description={"forward": bert_sequence_parallel_forward_fn(self.shard_config)},
                policy=policy,
                target_key=BertModel,
            )

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="word_embeddings",
                        target_module=embedding_cls,
                        kwargs=(
                            {
                                "fp8_communication": self.shard_config.fp8_communication,
                            }
                            if self.shard_config.enable_tensor_parallelism
                            else {}
                        ),
                    )
                ],
                policy=policy,
                target_key=BertEmbeddings,
            )

        # optimization configuration
        # Handle bert layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="attention.output.LayerNorm",
                    target_module=norm_cls,
                    kwargs={"sp_partial_derived": sp_partial_derived},
                ),
                SubModuleReplacementDescription(
                    suffix="output.LayerNorm",
                    target_module=norm_cls,
                    kwargs={"sp_partial_derived": sp_partial_derived},
                ),
            ],
            policy=policy,
            target_key=BertLayer,
        )
        # handle embedding layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="LayerNorm",
                    target_module=norm_cls,
                )
            ],
            policy=policy,
            target_key=BertEmbeddings,
        )

        # use jit operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_bert_self_output_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=BertSelfOutput,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_bert_output_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=BertOutput,
            )

        return policy

    def add_lm_head_policy(self, base_policy):
        from transformers.models.bert.modeling_bert import BertLMPredictionHead

        # optimize for tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="decoder",
                    target_module=col_nn.VocabParallelLMHead1D,
                    kwargs={
                        "gather_output": True,
                        "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by,
                        "fp8_communication": self.shard_config.fp8_communication,
                    },
                ),
                policy=base_policy,
                target_key=BertLMPredictionHead,
            )
        else:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="decoder",
                    target_module=col_nn.PaddingLMHead,
                    kwargs={"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by},
                ),
                policy=base_policy,
                target_key=BertLMPredictionHead,
            )

        # optimize with fused normalization
        if self.shard_config.enable_fused_normalization:
            # Handle bert lm prediction head
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="transform.LayerNorm",
                    target_module=col_nn.FusedLayerNorm,
                ),
                policy=base_policy,
                target_key=BertLMPredictionHead,
            )
        return base_policy

    def add_lm_prediction_policy(self, base_policy):
        from transformers.models.bert.modeling_bert import BertLMPredictionHead

        method_replacement = {
            "_save_to_state_dict": col_nn.ParallelModule._save_to_state_dict,
            "_load_from_state_dict": col_nn.ParallelModule._load_from_state_dict,
        }
        self.append_or_create_method_replacement(
            description=method_replacement,
            policy=base_policy,
            target_key=BertLMPredictionHead,
        )
        return base_policy

    def postprocess(self):
        return self.model

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """
        If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy.
        """
        if self.pipeline_stage_manager is None:
            return

        stage_manager = self.pipeline_stage_manager
        if self.model.__class__.__name__ == "BertModel":
            module = self.model
        else:
            module = self.model.bert

        if stage_manager.is_interleave:
            layers_per_stage = stage_manager.distribute_layers(len(module.encoder.layer))
            stage_manager.stage_indices = stage_manager.get_stage_index(layers_per_stage)
            method_replacement = {
                "forward": partial(
                    new_forward,
                    stage_manager=stage_manager,
                    shard_config=self.shard_config,
                )
            }

        else:
            layers_per_stage = stage_manager.distribute_layers(len(module.encoder.layer))
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

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "BertModel":
            module = self.model
        else:
            module = self.model.bert
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        if stage_manager.is_interleave:
            assert stage_manager.num_model_chunks is not None
            layers_per_stage = stage_manager.distribute_layers(len(module.encoder.layer))
            stage_indices = stage_manager.get_stage_index(layers_per_stage)
            if stage_manager.is_first_stage(ignore_chunk=True):
                held_layers.append(module.embeddings)
            for start_idx, end_idx in stage_indices:
                held_layers.extend(module.encoder.layer[start_idx:end_idx])
            if stage_manager.is_last_stage(ignore_chunk=True):
                held_layers.append(module.pooler)

        else:
            layers_per_stage = stage_manager.distribute_layers(len(module.encoder.layer))
            if stage_manager.is_first_stage():
                held_layers.append(module.embeddings)
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
            held_layers.extend(module.encoder.layer[start_idx:end_idx])
            if stage_manager.is_last_stage():
                held_layers.append(module.pooler)

        return held_layers


# BertModel
class BertModelPolicy(BertPolicy):
    def module_policy(self):
        policy = super().module_policy()
        from transformers.models.bert.modeling_bert import BertModel

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertModel,
                new_forward=BertPipelineForwards.bert_model_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        held_layers = super().get_held_layers()
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in bert model"""
        return []


# BertForPreTraining
class BertForPreTrainingPolicy(BertPolicy):
    def module_policy(self):
        policy = super().module_policy()
        policy = self.add_lm_head_policy(policy)
        policy = self.add_lm_prediction_policy(policy)
        from transformers.models.bert.modeling_bert import BertForPreTraining

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForPreTraining,
                new_forward=BertPipelineForwards.bert_for_pretraining_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage"""
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.cls)

        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        model = self.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if id(model.bert.embeddings.word_embeddings.weight) == id(model.cls.predictions.decoder.weight):
                # tie weights
                return [
                    {
                        0: model.bert.embeddings.word_embeddings.weight,
                        self.pipeline_stage_manager.num_stages - 1: model.cls.predictions.decoder.weight,
                    }
                ]
        return []


# BertLMHeadModel
class BertLMHeadModelPolicy(BertPolicy):
    def module_policy(self):
        policy = super().module_policy()
        policy = self.add_lm_head_policy(policy)
        policy = self.add_lm_prediction_policy(policy)
        from transformers.models.bert.modeling_bert import BertLMHeadModel

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertLMHeadModel,
                new_forward=BertPipelineForwards.bert_lm_head_model_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.cls)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        bert_model = self.model.bert
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if id(bert_model.embeddings.word_embeddings.weight) == id(self.model.cls.predictions.decoder.weight):
                # tie weights
                return [
                    {
                        0: bert_model.embeddings.word_embeddings.weight,
                        self.pipeline_stage_manager.num_stages - 1: self.model.cls.predictions.decoder.weight,
                    }
                ]
        return []


# BertForMaskedLM
class BertForMaskedLMPolicy(BertPolicy):
    def module_policy(self):
        policy = super().module_policy()
        policy = self.add_lm_head_policy(policy)
        policy = self.add_lm_prediction_policy(policy)
        from transformers.models.bert.modeling_bert import BertForMaskedLM

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForMaskedLM,
                new_forward=BertPipelineForwards.bert_for_masked_lm_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.cls)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        bert_model = self.model.bert
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if id(bert_model.embeddings.word_embeddings.weight) == id(self.model.cls.predictions.decoder.weight):
                # tie weights
                return [
                    {
                        0: bert_model.embeddings.word_embeddings.weight,
                        self.pipeline_stage_manager.num_stages - 1: self.model.cls.predictions.decoder.weight,
                    }
                ]
        return []


# BertForSequenceClassification
class BertForSequenceClassificationPolicy(BertPolicy):
    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForSequenceClassification

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                BertForSequenceClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        )
                    ]
                )
            }
            policy.update(addon_module)
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForSequenceClassification,
                new_forward=BertPipelineForwards.bert_for_sequence_classification_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.dropout)
            held_layers.append(self.model.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []


# BertForTokenClassification
class BertForTokenClassificationPolicy(BertPolicy):
    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForTokenClassification

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                BertForTokenClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        )
                    ]
                )
            }
            policy.update(addon_module)
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForTokenClassification,
                new_forward=BertPipelineForwards.bert_for_token_classification_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.dropout)
            held_layers.append(self.model.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []


# BertForNextSentencePrediction
class BertForNextSentencePredictionPolicy(BertPolicy):
    def module_policy(self):
        policy = super().module_policy()
        from transformers.models.bert.modeling_bert import BertForNextSentencePrediction

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForNextSentencePrediction,
                new_forward=BertPipelineForwards.bert_for_next_sentence_prediction_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.cls)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []


# BertForMultipleChoice
class BertForMultipleChoicePolicy(BertPolicy):
    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForMultipleChoice

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                BertForMultipleChoice: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        )
                    ]
                )
            }
            policy.update(addon_module)
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForMultipleChoice,
                new_forward=BertPipelineForwards.bert_for_multiple_choice_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.dropout)
            held_layers.append(self.model.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []


class BertForQuestionAnsweringPolicy(BertPolicy):
    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForQuestionAnswering

        policy = super().module_policy()
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForQuestionAnswering,
                new_forward=BertPipelineForwards.bert_for_question_answering_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.qa_outputs)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []
