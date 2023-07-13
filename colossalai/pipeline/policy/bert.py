from functools import partial
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.bert.modeling_bert import (
    BertForPreTraining,
    BertForPreTrainingOutput,
    BertLMHeadModel,
    BertModel,
)
from transformers.utils import ModelOutput, logging

from colossalai.pipeline.stage_manager import PipelineStageManager

from .base import Policy

logger = logging.get_logger(__name__)


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
    # labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,    # this is from the previous stage
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

    if token_type_ids is None:
        if hasattr(self.embeddings, "token_type_ids"):
            buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

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

    # calculate the num_layers
    num_layers_per_stage = len(self.encoder.layer) // stage_manager.num_stages
    start_layer = stage_manager.stage * num_layers_per_stage
    end_layer = (stage_manager.stage + 1) * num_layers_per_stage

    # layer_outputs
    layer_outputs = hidden_states if hidden_states is not None else None
    for idx, encoder_layer in enumerate(self.encoder.layer[start_layer:end_layer], start=start_layer):
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
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
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


# The layer partition policy for bertmodel
class BertModelPolicy(Policy):

    def __init__(
        self,
        stage_manager: PipelineStageManager,
        num_layers: int,
    ):
        super().__init__(stage_manager=stage_manager)
        self.stage_manager = stage_manager
        self.layers_per_stage = self.distribute_layers(num_layers, stage_manager.num_stages)

    def get_hold_layers(self, module: BertModel) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        hold_layers = []
        if self.stage_manager.is_first_stage():
            hold_layers.append(module.embeddings)
        start_idx, end_idx = self.get_stage_index(self.layers_per_stage, self.stage_manager.stage)
        hold_layers.extend(module.encoder.layer[start_idx:end_idx])
        if self.stage_manager.is_last_stage():
            hold_layers.append(module.pooler)

        return hold_layers

    def get_shared_params(self, module: BertModel) -> List[Dict[int, Tensor]]:
        '''no shared params in bertmodel'''
        return []

    def replace_forward(self, module: Module) -> None:
        module.forward = MethodType(partial(bert_model_forward, stage_manager=self.stage_manager), module)


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
                                 stage_manager=stage_manager,
                                 hidden_states=hidden_states if hidden_states is not None else None)
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


class BertForPreTrainingPolicy(Policy):

    def __init__(self, stage_manager: PipelineStageManager, num_layers: int):
        super().__init__(stage_manager=stage_manager)
        self.stage_manager = stage_manager
        self.layers_per_stage = self.distribute_layers(num_layers, stage_manager.num_stages)

    def get_hold_layers(self, module: BertForPreTraining) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        hold_layers = []
        if self.stage_manager.is_first_stage():
            hold_layers.append(module.bert.embeddings)

        start_idx, end_idx = self.get_stage_index(self.layers_per_stage, self.stage_manager.stage)
        hold_layers.extend(module.bert.encoder.layer[start_idx:end_idx])

        if self.stage_manager.is_last_stage():
            hold_layers.append(module.bert.pooler)
            hold_layers.append(module.cls)

        return hold_layers

    def get_shared_params(self, module: BertForPreTraining) -> List[Dict[int, Tensor]]:
        '''no shared params in bertmodel'''
        return []

    def replace_forward(self, module: Module) -> None:
        module.forward = MethodType(partial(bert_for_pretraining_forward, stage_manager=self.stage_manager),
                                    module.forward)


def bert_lmhead_forward(self: BertLMHeadModel,
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
                        stage_manager: Optional[PipelineStageManager] = None):
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
                                 hidden_states=hidden_states if hidden_states is not None else None)
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


class BertLMHeadModelPolicy(Policy):

    def __init__(self, stage_manager: PipelineStageManager, num_layers: int):
        super().__init__(stage_manager=stage_manager)
        self.stage_manager = stage_manager
        self.layers_per_stage = self.distribute_layers(num_layers, stage_manager.num_stages)

    def get_hold_layers(self, module: BertLMHeadModel) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        hold_layers = []
        if self.stage_manager.is_first_stage():
            hold_layers.append(module.bert.embeddings)
        start_idx, end_idx = self.get_stage_index(self.layers_per_stage, self.stage_manager.stage)
        hold_layers.extend(module.bert.encoder.layer[start_idx:end_idx])
        if self.stage_manager.is_last_stage():
            hold_layers.append(module.bert.pooler)
            hold_layers.append(module.cls)

        return hold_layers

    def get_shared_params(self, module: BertLMHeadModel) -> List[Dict[int, Tensor]]:
        '''no shared params in bertmodel'''
        return []

    def replace_forward(self, module: Module) -> None:
        module.forward = MethodType(partial(bert_lmhead_forward, stage_manager=self.stage_manager), module)
