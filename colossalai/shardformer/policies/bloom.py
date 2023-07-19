import warnings
from functools import partial
from types import MethodType
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.models.bloom.modeling_bloom import (
    BloomForCausalLM,
    BloomForQuestionAnswering,
    BloomForSequenceClassification,
    BloomForTokenClassification,
    BloomModel,
)
from transformers.utils import logging

import colossalai.shardformer.layer as col_nn
from colossalai.pipeline.stage_manager import PipelineStageManager

from .._utils import getattr_, setattr_
from ..modeling.bloom import build_bloom_alibi_tensor_fn
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

logger = logging.get_logger(__name__)


class BloomPolicy(Policy):

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
        from transformers.models.bloom.modeling_bloom import BloomBlock, BloomModel

        policy = {}

        if self.shard_config.enable_tensor_parallelism:
            policy[BloomBlock] = ModulePolicyDescription(attribute_replacement={
                "self_attention.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attention.split_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attention.num_heads": self.model.config.n_head // self.shard_config.tensor_parallel_size,
            },
                                                         sub_module_replacement=[
                                                             SubModuleReplacementDescription(
                                                                 suffix="self_attention.query_key_value",
                                                                 target_module=col_nn.Linear1D_Col,
                                                             ),
                                                             SubModuleReplacementDescription(
                                                                 suffix="self_attention.dense",
                                                                 target_module=col_nn.Linear1D_Row,
                                                             ),
                                                             SubModuleReplacementDescription(
                                                                 suffix="self_attention.attention_dropout",
                                                                 target_module=col_nn.DropoutForParallelInput,
                                                             ),
                                                             SubModuleReplacementDescription(
                                                                 suffix="mlp.dense_h_to_4h",
                                                                 target_module=col_nn.Linear1D_Col,
                                                             ),
                                                             SubModuleReplacementDescription(
                                                                 suffix="mlp.dense_4h_to_h",
                                                                 target_module=col_nn.Linear1D_Row,
                                                             ),
                                                         ])

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
                ])

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            # handle bloom model
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(
                    suffix="ln_f",
                    target_module=col_nn.FusedLayerNorm,
                ),
                SubModuleReplacementDescription(
                    suffix="word_embeddings_layernorm",
                    target_module=col_nn.FusedLayerNorm,
                )
            ],
                                                        policy=policy,
                                                        target_key=BloomModel)

            # handle bloom block
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(
                    suffix="input_layernorm",
                    target_module=col_nn.FusedLayerNorm,
                ),
                SubModuleReplacementDescription(
                    suffix="post_attention_layernorm",
                    target_module=col_nn.FusedLayerNorm,
                )
            ],
                                                        policy=policy,
                                                        target_key=BloomBlock)

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
            method_replacement = {'forward': partial(new_forward, stage_manager=stage_manager, stage_index=stage_index)}
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=model_cls)
        return


class BloomModelPolicy(BloomPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        from transformers.models.bloom.modeling_bloom import BloomModel
        self.set_pipeline_forward(model_cls=BloomModel, new_forward=bloom_model_forward, policy=policy)
        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        module = self.model
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

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        '''no shared params in bloom model'''
        return []


class BloomForCausalLMPolicy(BloomPolicy):

    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomForCausalLM
        policy = super().module_policy()

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="lm_head", target_module=col_nn.Linear1D_Col, kwargs=dict(gather_output=True)),
                                                        policy=policy,
                                                        target_key=BloomForCausalLM)

        self.set_pipeline_forward(model_cls=BloomForCausalLM, new_forward=bloom_for_causal_lm_forward, policy=policy)
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.transformer.h), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.transformer.word_embeddings)
            held_layers.append(module.transformer.word_embeddings_layernorm)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.transformer.h[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.transformer.ln_f)
            held_layers.append(module.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        bloom_model = self.model
        if self.pipeline_stage_manager:
            if id(bloom_model.transformer.word_embeddings.weight) == id(bloom_model.lm_head.weight):
                # tie weights
                return [{
                    0: bloom_model.transformer.word_embeddings.weight,
                    self.stage_manager.num_stages - 1: bloom_model.lm_head.weight
                }]
        return []

    def postprocess(self):
        if self.shard_config.enable_tensor_parallelism and self.pipeline_stage_manager is None:
            binding_map = {"transformer.word_embeddings.weight": "lm_head.weight"}

            for k, v in binding_map.items():
                param = getattr_(self.model, k)
                # tie weights
                setattr_(self.model, v, param)
        return self.model


class BloomForSequenceClassificationPolicy(BloomPolicy):

    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomForSequenceClassification
        policy = super().module_policy()

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="score", target_module=col_nn.Linear1D_Col, kwargs=dict(gather_output=True)),
                                                        policy=policy,
                                                        target_key=BloomForSequenceClassification)
        self.set_pipeline_forward(model_cls=BloomForSequenceClassification,
                                  new_forward=bloom_for_sequence_classification_forward,
                                  policy=policy)
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.transformer.h), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.transformer.word_embeddings)
            held_layers.append(module.transformer.word_embeddings_layernorm)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.transformer.h[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.transformer.ln_f)
            held_layers.append(module.score)
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
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(suffix="classifier",
                                                target_module=col_nn.Linear1D_Col,
                                                kwargs=dict(gather_output=True)),
                SubModuleReplacementDescription(
                    suffix="dropout",
                    target_module=col_nn.DropoutForReplicatedInput,
                ),
            ],
                                                        policy=policy,
                                                        target_key=BloomForTokenClassification)

        self.set_pipeline_forward(model_cls=BloomForTokenClassification,
                                  new_forward=bloom_for_token_classification_forward,
                                  policy=policy)

        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.transformer.h), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.transformer.word_embeddings)
            held_layers.append(module.transformer.word_embeddings_layernorm)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.transformer.h[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.transformer.ln_f)
            held_layers.append(module.dropout)
            held_layers.append(module.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in bloom for token classification model"""
        return []


class BloomForQuestionAnsweringPolicy(BloomPolicy):
    # No head sharding as the output features is only 2
    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomForQuestionAnswering
        policy = super().module_policy()
        self.set_pipeline_forward(model_cls=BloomForQuestionAnswering,
                                  new_forward=bloom_for_question_answering_forward,
                                  policy=policy)
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.transformer.h), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.transformer.word_embeddings)
            held_layers.append(module.transformer.word_embeddings_layernorm)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.transformer.h[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.transformer.ln_f)
            held_layers.append(module.qa_outputs)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in bloom for question answering model"""
        return []


def bloom_model_forward(
    self: BloomModel,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    stage_index: Optional[List[int]] = None,
    **deprecated_arguments,
) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
    if deprecated_arguments.pop("position_ids", False) is not False:
        # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
        warnings.warn(
            "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
            " passing `position_ids`.",
            FutureWarning,
        )
    if len(deprecated_arguments) > 0:
        raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (output_hidden_states
                            if output_hidden_states is not None else self.config.output_hidden_states)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # add warnings here
    if output_attentions:
        logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
        output_attentions = False
    if output_hidden_states:
        logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
        output_hidden_states = False
    if use_cache:
        logger.warning_once('use_cache=True is not supported for pipeline models at the moment.')
        use_cache = False
    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape batch_size x num_heads x N x N

    # head_mask has shape n_layer x batch x num_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    # case: First stage of training
    if stage_manager.is_first_stage():
        # check input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        # initialize in the first stage and then pass to the next stage
    else:
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_length = input_shape

    # extra recording tensor should be generated in the first stage

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

    if past_key_values is None:
        past_key_values = tuple([None] * len(self.h))
    # Compute alibi tensor: check build_alibi_tensor documentation,build for every stage
    seq_length_with_past = seq_length
    past_key_values_length = 0
    if past_key_values[0] is not None:
        past_key_values_length = past_key_values[0][0].shape[2]    # source_len

        seq_length_with_past = seq_length_with_past + past_key_values_length
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
    else:
        attention_mask = attention_mask.to(hidden_states.device)

    alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

    # causal_mask is constructed every stage and its input is passed through different stages
    causal_mask = self._prepare_attn_mask(
        attention_mask,
        input_shape=(batch_size, seq_length),
        past_key_values_length=past_key_values_length,
    )

    start_idx, end_idx = stage_index[0], stage_index[1]
    for i, (block, layer_past) in enumerate(zip(self.h[start_idx:end_idx], past_key_values[start_idx:end_idx])):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):

                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                alibi,
                causal_mask,
                layer_past,
                head_mask[i],
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
            )

        hidden_states = outputs[0]

        if use_cache is True:
            presents = presents + (outputs[1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + \
                (outputs[2 if use_cache else 1],)

    if stage_manager.is_last_stage():
        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

    # TODO: deal with all_hidden_states, all_self_attentions, presents
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if stage_manager.is_last_stage():
        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        # attention_mask is not returned ; presents = past_key_values
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    else:
        # always return dict for imediate stage
        return {'hidden_states': hidden_states}


def bloom_for_causal_lm_forward(self: 'BloomForCausalLM',
                                input_ids: Optional[torch.LongTensor] = None,
                                past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                                attention_mask: Optional[torch.Tensor] = None,
                                head_mask: Optional[torch.Tensor] = None,
                                inputs_embeds: Optional[torch.Tensor] = None,
                                labels: Optional[torch.Tensor] = None,
                                use_cache: Optional[bool] = None,
                                output_attentions: Optional[bool] = None,
                                output_hidden_states: Optional[bool] = None,
                                return_dict: Optional[bool] = None,
                                stage_manager: Optional[PipelineStageManager] = None,
                                hidden_states: Optional[torch.FloatTensor] = None,
                                stage_index: Optional[List[int]] = None,
                                **deprecated_arguments):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
    """
    if deprecated_arguments.pop("position_ids", False) is not False:
        # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
        warnings.warn(
            "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
            " passing `position_ids`.",
            FutureWarning,
        )
    if len(deprecated_arguments) > 0:
        raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

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

    transformer_outputs = bloom_model_forward(self.transformer,
                                              input_ids,
                                              past_key_values=past_key_values,
                                              attention_mask=attention_mask,
                                              head_mask=head_mask,
                                              inputs_embeds=inputs_embeds,
                                              use_cache=use_cache,
                                              output_attentions=output_attentions,
                                              output_hidden_states=output_hidden_states,
                                              return_dict=return_dict,
                                              stage_manager=stage_manager,
                                              hidden_states=hidden_states,
                                              stage_index=stage_index)
    past_key_values = None
    all_hidden_states = None
    all_self_attentions = None
    all_cross_attentions = None
    if stage_manager.is_last_stage():
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(batch_size * seq_length, vocab_size),
                            shift_labels.view(batch_size * seq_length))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    else:
        hidden_states = transformer_outputs.get('hidden_states')
        return {'hidden_states': hidden_states}


def bloom_for_sequence_classification_forward(
    self: BloomForSequenceClassification,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    stage_index: Optional[List[int]] = None,
    **deprecated_arguments,
):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    if deprecated_arguments.pop("position_ids", False) is not False:
        # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
        warnings.warn(
            "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
            " passing `position_ids`.",
            FutureWarning,
        )
    if len(deprecated_arguments) > 0:
        raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

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

    transformer_outputs = bloom_model_forward(
        self.transformer,
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        stage_manager=stage_manager,
        hidden_states=hidden_states,
        stage_index=stage_index,
    )
    past_key_values = None
    all_hidden_states = None
    all_self_attentions = None
    all_cross_attentions = None
    if stage_manager.is_last_stage():
        batch_size = hidden_states.shape[0]
        #update batch size
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`")

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

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
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    else:
        hidden_states = transformer_outputs.get('hidden_states')
        return {'hidden_states': hidden_states}


def bloom_for_token_classification_forward(
    self: BloomForTokenClassification,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    stage_index: Optional[List[int]] = None,
    **deprecated_arguments,
):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    if deprecated_arguments.pop("position_ids", False) is not False:
        # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
        warnings.warn(
            "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
            " passing `position_ids`.",
            FutureWarning,
        )
    if len(deprecated_arguments) > 0:
        raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

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

    transformer_outputs = bloom_model_forward(
        self.transformer,
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        stage_manager=stage_manager,
        hidden_states=hidden_states,
        stage_index=stage_index,
    )
    past_key_values = None
    all_hidden_states = None
    all_self_attentions = None
    all_cross_attentions = None
    if stage_manager.is_last_stage():
        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            batch_size, seq_length = labels.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    else:
        hidden_states = transformer_outputs.get('hidden_states')
        return {'hidden_states': hidden_states}


def bloom_for_question_answering_forward(
    self: BloomForQuestionAnswering,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    start_positions: Optional[torch.LongTensor] = None,
    end_positions: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    stage_index: Optional[List[int]] = None,
):
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

    outputs = bloom_model_forward(
        self.transformer,
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        stage_manager=stage_manager,
        hidden_states=hidden_states,
        stage_index=stage_index,
    )
    past_key_values = None
    all_hidden_states = None
    all_self_attentions = None
    all_cross_attentions = None

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
