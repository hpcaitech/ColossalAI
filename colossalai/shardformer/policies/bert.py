from functools import partial
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.models.bert.modeling_bert import (
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForPreTrainingOutput,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertLMHeadModel,
    BertModel,
)
from transformers.utils import logging

import colossalai.shardformer.layer as col_nn
from colossalai.pipeline.stage_manager import PipelineStageManager

from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

logger = logging.get_logger(__name__)

__all__ = [
    'BertPolicy', 'BertModelPolicy', 'BertForPreTrainingPolicy', 'BertLMdHeadModelPolicy', 'BertForMaskedLMPolicy',
    'BertForNextSentencePredictionPolicy', 'BertForSequenceClassificationPolicy', 'BertForTokenClassificationPolicy',
    'BertForMultipleChoicePolicy', 'BertForQuestionAnsweringPolicy'
]


class BertPolicy(Policy):

    def config_sanity_check(self):
        pass

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        # TODO:
        if self.shard_config.enable_tensor_parallelism:
            vocab_size = self.model.config.vocab_size
            world_size = self.shard_config.tensor_parallel_size
            if vocab_size % world_size != 0:
                new_vocab_size = vocab_size + world_size - vocab_size % world_size
                self.model.resize_token_embeddings(new_vocab_size)
        return self.model

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertEmbeddings, BertLayer

        policy = {}

        if self.shard_config.enable_tensor_parallelism:
            policy[BertLayer] = ModulePolicyDescription(attribute_replacement={
                "attention.self.all_head_size":
                    self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "crossattention.self.all_head_size":
                    self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "attention.self.num_attention_heads":
                    self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                "crossattention.self.num_attention_heads":
                    self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
            },
                                                        sub_module_replacement=[
                                                            SubModuleReplacementDescription(
                                                                suffix="attention.self.query",
                                                                target_module=col_nn.Linear1D_Col,
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="attention.self.key",
                                                                target_module=col_nn.Linear1D_Col,
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="attention.self.value",
                                                                target_module=col_nn.Linear1D_Col,
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="attention.self.dropout",
                                                                target_module=col_nn.DropoutForParallelInput,
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="attention.output.dense",
                                                                target_module=col_nn.Linear1D_Row,
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="attention.output.dropout",
                                                                target_module=col_nn.DropoutForParallelInput,
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="intermediate.dense",
                                                                target_module=col_nn.Linear1D_Col,
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="output.dense",
                                                                target_module=col_nn.Linear1D_Row,
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="output.dropout",
                                                                target_module=col_nn.DropoutForParallelInput,
                                                            )
                                                        ])

            policy[BertEmbeddings] = ModulePolicyDescription(sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="word_embeddings",
                    target_module=col_nn.VocabParallelEmbedding1D,
                ),
                SubModuleReplacementDescription(
                    suffix="dropout",
                    target_module=col_nn.DropoutForReplicatedInput,
                )
            ])

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            # Handle bert layer
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(
                    suffix="attention.output.LayerNorm",
                    target_module=col_nn.FusedLayerNorm,
                ),
                SubModuleReplacementDescription(
                    suffix="output.LayerNorm",
                    target_module=col_nn.FusedLayerNorm,
                )
            ],
                                                        policy=policy,
                                                        target_key=BertLayer)
            # handle embedding layer
            self.append_or_create_submodule_replacement(
                description=[SubModuleReplacementDescription(
                    suffix="LayerNorm",
                    target_module=col_nn.FusedLayerNorm,
                )],
                policy=policy,
                target_key=BertEmbeddings)

        return policy

    def add_lm_head_policy(self, base_policy):
        from transformers.models.bert.modeling_bert import BertLMPredictionHead

        # optimize for tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="decoder", target_module=col_nn.Linear1D_Col, kwargs={"gather_output": True}),
                                                        policy=base_policy,
                                                        target_key=BertLMPredictionHead)

        # optimize with fused normalization
        if self.shard_config.enable_fused_normalization:
            # Handle bert lm prediction head
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="transform.LayerNorm",
                target_module=col_nn.FusedLayerNorm,
            ),
                                                        policy=base_policy,
                                                        target_key=BertLMPredictionHead)
        return base_policy

    def add_lm_prediction_policy(self, base_policy):
        from transformers.models.bert.modeling_bert import BertLMPredictionHead
        method_replacement = {
            '_save_to_state_dict': col_nn.ParallelModule._save_to_state_dict,
            '_load_from_state_dict': col_nn.ParallelModule._load_from_state_dict,
        }
        self.append_or_create_method_replacement(description=method_replacement,
                                                 policy=base_policy,
                                                 target_key=BertLMPredictionHead)
        return base_policy

    def postprocess(self):
        return self.model

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
           to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == "BertModel":
                module = self.model
            else:
                module = self.model.bert

            layers_per_stage = Policy.distribute_layers(len(module.encoder.layer), stage_manager.num_stages)
            stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
            method_replacement = {'forward': partial(new_forward, stage_manager=stage_manager, stage_index=stage_index)}
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=model_cls)

        return


# BertModel
class BertModelPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        from transformers.models.bert.modeling_bert import BertModel
        self.set_pipeline_forward(model_cls=BertModel, new_forward=bert_model_forward, policy=policy)
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.encoder.layer), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.embeddings)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.encoder.layer[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.pooler)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in bert model"""
        return []


# BertForPreTraining
class BertForPreTrainingPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        policy = self.add_lm_head_policy(policy)
        policy = self.add_lm_prediction_policy(policy)
        from transformers.models.bert.modeling_bert import BertForPreTraining
        self.set_pipeline_forward(model_cls=BertForPreTraining, new_forward=bert_for_pretraining_forward, policy=policy)
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage"""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        layers_per_stage = self.distribute_layers(len(module.bert.encoder.layer), stage_manager.num_stages)
        held_layers = []
        if stage_manager.is_first_stage():
            held_layers.append(module.bert.embeddings)

        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.bert.encoder.layer[start_idx:end_idx])

        if stage_manager.is_last_stage():
            held_layers.append(module.bert.pooler)
            held_layers.append(module.cls)

        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        model = self.model
        if self.pipeline_stage_manager:
            if id(model.bert.embeddings.word_embeddings.weight) == id(model.cls.predictions.decoder.weight):
                # tie weights
                return [{
                    0: model.bert.embeddings.word_embeddings.weight,
                    self.pipeline_stage_manager.num_stages - 1: model.cls.predictions.decoder.weight
                }]
        return []


# BertLMHeadModel
class BertLMHeadModelPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        policy = self.add_lm_head_policy(policy)
        policy = self.add_lm_prediction_policy(policy)
        from transformers.models.bert.modeling_bert import BertLMHeadModel
        self.set_pipeline_forward(model_cls=BertLMHeadModel, new_forward=bert_lm_head_model_forward, policy=policy)
        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        module = self.model
        held_layers = []
        stage_manager = self.pipeline_stage_manager
        layers_per_stage = self.distribute_layers(len(module.bert.encoder.layer), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.bert.embeddings)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.bert.encoder.layer[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.bert.pooler)
            held_layers.append(module.cls)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        bert_model = self.model.bert
        if self.pipeline_stage_manager:
            if id(bert_model.embeddings.word_embeddings.weight) == id(self.model.cls.predictions.decoder.weight):
                # tie weights
                return [{
                    0: bert_model.embeddings.word_embeddings.weight,
                    self.pipeline_stage_manager.num_stages - 1: self.model.cls.predictions.decoder.weight
                }]
        return []


# BertForMaskedLM
class BertForMaskedLMPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        policy = self.add_lm_head_policy(policy)
        mpolicy = self.add_lm_prediction_policy(policy)
        from transformers.models.bert.modeling_bert import BertForMaskedLM
        self.set_pipeline_forward(model_cls=BertForMaskedLM, new_forward=bert_for_masked_lm_forward, policy=policy)
        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        module = self.model
        held_layers = []
        stage_manager = self.pipeline_stage_manager
        layers_per_stage = self.distribute_layers(len(module.bert.encoder.layer), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.bert.embeddings)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.bert.encoder.layer[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.bert.pooler)
            held_layers.append(module.cls)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        bert_model = self.model.bert
        if self.pipeline_stage_manager:
            if id(bert_model.embeddings.word_embeddings.weight) == id(self.model.cls.predictions.decoder.weight):
                # tie weights
                return [{
                    0: bert_model.embeddings.word_embeddings.weight,
                    self.pipeline_stage_manager.num_stages - 1: self.model.cls.predictions.decoder.weight
                }]
        return []


# BertForSequenceClassification
class BertForSequenceClassificationPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForSequenceClassification

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                BertForSequenceClassification:
                    ModulePolicyDescription(sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        )
                    ])
            }
            policy.update(addon_module)

        self.set_pipeline_forward(model_cls=BertForSequenceClassification,
                                  new_forward=bert_for_sequence_classification_forward,
                                  policy=policy)

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        module = self.model
        held_layers = []
        stage_manager = self.pipeline_stage_manager
        layers_per_stage = self.distribute_layers(len(module.bert.encoder.layer), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.bert.embeddings)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.bert.encoder.layer[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.bert.pooler)
            held_layers.append(module.dropout)
            held_layers.append(module.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []


# BertForTokenClassification
class BertForTokenClassificationPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForTokenClassification

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                BertForTokenClassification:
                    ModulePolicyDescription(sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        )
                    ])
            }
            policy.update(addon_module)

        self.set_pipeline_forward(model_cls=BertForTokenClassification,
                                  new_forward=bert_for_token_classification_forward,
                                  policy=policy)

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        module = self.model
        held_layers = []
        stage_manager = self.pipeline_stage_manager
        layers_per_stage = self.distribute_layers(len(module.bert.encoder.layer), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.bert.embeddings)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.bert.encoder.layer[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.bert.pooler)
            held_layers.append(module.dropout)
            held_layers.append(module.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []


# BertForNextSentencePrediction
class BertForNextSentencePredictionPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        from transformers.models.bert.modeling_bert import BertForNextSentencePrediction
        self.set_pipeline_forward(model_cls=BertForNextSentencePrediction,
                                  new_forward=bert_for_next_sentence_prediction_forward,
                                  policy=policy)

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        module = self.model
        held_layers = []
        stage_manager = self.pipeline_stage_manager
        layers_per_stage = self.distribute_layers(len(module.bert.encoder.layer), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.bert.embeddings)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.bert.encoder.layer[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.bert.pooler)
            held_layers.append(module.cls)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []


# BertForMultipleChoice
class BertForMultipleChoicePolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForMultipleChoice

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                BertForMultipleChoice:
                    ModulePolicyDescription(sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        )
                    ])
            }
            policy.update(addon_module)

        self.set_pipeline_forward(model_cls=BertForMultipleChoice,
                                  new_forward=bert_for_multiple_choice_forward,
                                  policy=policy)

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        module = self.model
        held_layers = []
        stage_manager = self.pipeline_stage_manager
        layers_per_stage = self.distribute_layers(len(module.bert.encoder.layer), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.bert.embeddings)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.bert.encoder.layer[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.bert.pooler)
            held_layers.append(module.dropout)
            held_layers.append(module.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []


class BertForQuestionAnsweringPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForQuestionAnswering
        policy = super().module_policy()
        self.set_pipeline_forward(model_cls=BertForQuestionAnswering,
                                  new_forward=bert_for_question_answering_forward,
                                  policy=policy)

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        module = self.model
        held_layers = []
        stage_manager = self.pipeline_stage_manager
        layers_per_stage = self.distribute_layers(len(module.bert.encoder.layer), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.bert.embeddings)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.bert.encoder.layer[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.bert.pooler)
            held_layers.append(module.qa_outputs)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []


def bert_model_forward(
    self: BertModel,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    hidden_states: Optional[torch.FloatTensor] = None,    # this is from the previous stage
    stage_index: Optional[List[int]] = None,
):
    # TODO: add explaination of the output here.
    r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
    # debugging
    # preprocess:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (output_hidden_states
                            if output_hidden_states is not None else self.config.output_hidden_states)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if self.config.is_decoder:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
    else:
        use_cache = False

    if stage_manager.is_first_stage():
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    else:
        input_shape = hidden_states.size()[:-1]
        batch_size, seq_length = input_shape
        device = hidden_states.device

    # TODO: left the recording kv-value tensors as () or None type, this feature may be added in the future.
    if output_attentions:
        logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
        output_attentions = False
    if output_hidden_states:
        logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
        output_hidden_states = False
    if use_cache:
        logger.warning_once('use_cache=True is not supported for pipeline models at the moment.')
        use_cache = False

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    if attention_mask is None:
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
    attention_mask = extended_attention_mask
    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    hidden_states = hidden_states if hidden_states is not None else None

    if stage_manager.is_first_stage():
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

    # inherit from bert_layer,this should be changed when we add the feature to record hidden_states
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

    if self.encoder.gradient_checkpointing and self.encoder.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False
    next_decoder_cache = () if use_cache else None

    start_idx, end_idx = stage_index[0], stage_index[1]
    # layer_outputs
    layer_outputs = hidden_states if hidden_states is not None else None
    for idx, encoder_layer in enumerate(self.encoder.layer[start_idx:end_idx], start=start_idx):
        if stage_manager.is_first_stage() and idx == 0:
            encoder_attention_mask = encoder_extended_attention_mask

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[idx] if head_mask is not None else None
        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.encoder.gradient_checkpointing and self.encoder.training:

            def create_custom_forward(module):

                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(encoder_layer),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + \
                    (layer_outputs[2],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    # end of a stage loop
    sequence_output = layer_outputs[0] if layer_outputs is not None else None

    if stage_manager.is_last_stage():
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + layer_outputs[1:]
        # return dict is not supported at this moment
        else:
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )

    # output of non-first and non-last stages: must be a dict
    else:
        # intermediate stage always return dict
        return {
            'hidden_states': hidden_states,
        }


def bert_for_pretraining_forward(
    self: BertForPreTraining,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    next_sentence_label: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    stage_index: Optional[List[int]] = None,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    # TODO: left the recording kv-value tensors as () or None type, this feature may be added in the future.
    if output_attentions:
        logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
        output_attentions = False
    if output_hidden_states:
        logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
        output_hidden_states = False
    if return_dict:
        logger.warning_once('return_dict is not supported for pipeline models at the moment')
        return_dict = False

    outputs = bert_model_forward(
        self.bert,
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        stage_manager=stage_manager,
        hidden_states=hidden_states if hidden_states is not None else None,
        stage_index=stage_index,
    )
    past_key_values = None
    all_hidden_states = None
    all_self_attentions = None
    all_cross_attentions = None
    if stage_manager.is_last_stage():
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        # the last stage for pretraining model
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    else:
        hidden_states = outputs.get('hidden_states')

        # intermediate stage always return dict
        return {
            'hidden_states': hidden_states,
        }


def bert_lm_head_model_forward(
    self: BertLMHeadModel,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.Tensor]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    stage_index: Optional[List[int]] = None,
):
    r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if labels is not None:
        use_cache = False
    if output_attentions:
        logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
        output_attentions = False
    if output_hidden_states:
        logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
        output_hidden_states = False
    if return_dict:
        logger.warning_once('return_dict is not supported for pipeline models at the moment')
        return_dict = False

    outputs = bert_model_forward(self.bert,
                                 input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds,
                                 encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_attention_mask,
                                 past_key_values=past_key_values,
                                 use_cache=use_cache,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states,
                                 return_dict=return_dict,
                                 stage_manager=stage_manager,
                                 hidden_states=hidden_states if hidden_states is not None else None,
                                 stage_index=stage_index)
    past_key_values = None
    all_hidden_states = None
    all_self_attentions = None
    all_cross_attentions = None

    if stage_manager.is_last_stage():
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
    else:
        hidden_states = outputs.get('hidden_states')
        # intermediate stage always return dict
        return {'hidden_states': hidden_states}


def bert_for_masked_lm_forward(
    self: BertForMaskedLM,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    hidden_states: Optional[torch.Tensor] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    stage_index: Optional[List[int]] = None,
):
    # -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if output_attentions:
        logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
        output_attentions = False
    if output_hidden_states:
        logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
        output_hidden_states = False
    if return_dict:
        logger.warning_once('return_dict is not supported for pipeline models at the moment')
        return_dict = False

    outputs = bert_model_forward(
        self.bert,
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        hidden_states=hidden_states,
        stage_manager=stage_manager,
        stage_index=stage_index,
    )

    if stage_manager.is_last_stage():
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()    # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    else:
        hidden_states = outputs.get('hidden_states')
        return {'hidden_states': hidden_states}


def bert_for_next_sentence_prediction_forward(
    self: BertForNextSentencePrediction,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    hidden_states: Optional[torch.Tensor] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    stage_index: Optional[List[int]] = None,
    **kwargs,
):
    # -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

    if "next_sentence_label" in kwargs:
        warnings.warn(
            "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
            " `labels` instead.",
            FutureWarning,
        )
        labels = kwargs.pop("next_sentence_label")
    if output_attentions:
        logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
        output_attentions = False
    if output_hidden_states:
        logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
        output_hidden_states = False
    if return_dict:
        logger.warning_once('return_dict is not supported for pipeline models at the moment')
        return_dict = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = bert_model_forward(self.bert,
                                 input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states,
                                 return_dict=return_dict,
                                 hidden_states=hidden_states,
                                 stage_manager=stage_manager,
                                 stage_index=stage_index)

    if stage_manager.is_last_stage():
        pooled_output = outputs[1]
        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    else:
        hidden_states = outputs.get('hidden_states')
        # intermediate stage always return dict
        return {'hidden_states': hidden_states}


def bert_for_sequence_classification_forward(
    self: BertForSequenceClassification,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    hidden_states: Optional[torch.Tensor] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    stage_index: Optional[List[int]] = None,
):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if output_attentions:
        logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
        output_attentions = False
    if output_hidden_states:
        logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
        output_hidden_states = False
    if return_dict:
        logger.warning_once('return_dict is not supported for pipeline models at the moment')
        return_dict = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = bert_model_forward(self.bert,
                                 input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states,
                                 return_dict=return_dict,
                                 hidden_states=hidden_states,
                                 stage_manager=stage_manager,
                                 stage_index=stage_index)

    if stage_manager.is_last_stage():
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    else:
        hidden_states = outputs.get('hidden_states')
        return {'hidden_states': hidden_states}


def bert_for_token_classification_forward(
    self: BertForTokenClassification,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    hidden_states: Optional[torch.Tensor] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    stage_index: Optional[List[int]] = None,
):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if output_attentions:
        logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
        output_attentions = False
    if output_hidden_states:
        logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
        output_hidden_states = False
    if return_dict:
        logger.warning_once('return_dict is not supported for pipeline models at the moment')
        return_dict = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = bert_model_forward(
        self.bert,
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        hidden_states=hidden_states,
        stage_manager=stage_manager,
        stage_index=stage_index,
    )

    if stage_manager.is_last_stage():
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    else:
        hidden_states = outputs.get('hidden_states')
        return {'hidden_states': hidden_states}


def bert_for_multiple_choice_forward(
    self: BertForMultipleChoice,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    hidden_states: Optional[torch.Tensor] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    stage_index: Optional[List[int]] = None,
):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
        num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
        `input_ids` above)
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if output_attentions:
        logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
        output_attentions = False
    if output_hidden_states:
        logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
        output_hidden_states = False
    if return_dict:
        logger.warning_once('return_dict is not supported for pipeline models at the moment')
        return_dict = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # in our pipeline design,input ids are copied for every stage and shouldn't be none
    # the input_ids for multiple choice model is [batch_size, num_choices, sequence_length]
    if stage_manager.is_last_stage():
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

    input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
    attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
    token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
    position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
    inputs_embeds = (inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
                     if inputs_embeds is not None else None)

    outputs = bert_model_forward(
        self.bert,
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        hidden_states=hidden_states,
        stage_manager=stage_manager,
        stage_index=stage_index,
    )
    if stage_manager.is_last_stage():
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    else:
        hidden_states = outputs.get('hidden_states')
        return {'hidden_states': hidden_states}


def bert_for_question_answering_forward(
    self: BertForQuestionAnswering,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    start_positions: Optional[torch.Tensor] = None,
    end_positions: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    hidden_states: Optional[torch.Tensor] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    stage_index: Optional[List[int]] = None,
):
    # NOTE: the arg start_position and end_position are used only for the last stage
    r"""
    start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for position (index) of the start of the labelled span for computing the token classification loss.
        Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
        are not taken into account for computing the loss.
    end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for position (index) of the end of the labelled span for computing the token classification loss.
        Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
        are not taken into account for computing the loss.
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if output_attentions:
        logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
        output_attentions = False
    if output_hidden_states:
        logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
        output_hidden_states = False
    if return_dict:
        logger.warning_once('return_dict is not supported for pipeline models at the moment')
        return_dict = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = bert_model_forward(
        self.bert,
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        hidden_states=hidden_states,
        stage_manager=stage_manager,
        stage_index=stage_index,
    )
    if stage_manager.is_last_stage():
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    else:
        hidden_states = outputs.get('hidden_states')
        return {'hidden_states': hidden_states}
