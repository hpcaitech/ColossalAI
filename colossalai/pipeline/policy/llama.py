from functools import partial
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutput,
                                           CausalLMOutputWithPast)
from transformers.models.llama.modeling_llama import (LlamaForCausalLM,
                                                      LlamaModel)
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager

from .base import Policy

logger = logging.get_logger(__name__)


def llama_model_forward(self: LlamaModel,
                        input_ids: torch.LongTensor = None,
                        attention_mask: Optional[torch.Tensor] = None,
                        position_ids: Optional[torch.LongTensor] = None,
                        past_key_values: Optional[List[torch.FloatTensor]] = None,
                        inputs_embeds: Optional[torch.FloatTensor] = None,
                        labels: Optional[torch.LongTensor] = None,
                        use_cache: Optional[bool] = None,
                        output_attentions: Optional[bool] = None,
                        output_hidden_states: Optional[bool] = None,
                        return_dict: Optional[bool] = None,
                        stage_manager: Optional[PipelineStageManager] = None,  # this is set by partial
                        hidden_states: Optional[torch.FloatTensor] = None,  # this is from the previous stage
                        ) -> Union[CausalLMOutput, Tuple]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if output_attentions:
        logger.warning_once('`output_attentions=True` is not supported for pipeline models at the moment.')
        output_attentions = False
    if output_hidden_states:
        logger.warning_once('`output_hidden_states=True` is not supported for pipeline models at the moment.')
        output_hidden_states = False
    if use_cache:
        logger.warning_once('`use_cache=True` is not supported for pipeline models at the moment.')
        use_cache = False

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        if stage_manager.is_first_stage():
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            inputs_embeds = hidden_states
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
    # this function only uses inputs_embeds' device, dtype, and shape, it's safe to use hidden_state
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    num_layers_per_stage = len(self.layers) // stage_manager.num_stages
    start_layer = stage_manager.stage * num_layers_per_stage
    end_layer = (stage_manager.stage + 1) * num_layers_per_stage

    for idx, decoder_layer in enumerate(self.layers[start_layer:end_layer], start=start_layer):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    if stage_manager.is_last_stage():
        hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    # TODO(ver217): return_dict is not supported for pipeline models at the moment.
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def llama_for_causal_lm_forward(self: LlamaForCausalLM,
                                input_ids: torch.LongTensor = None,
                                attention_mask: Optional[torch.Tensor] = None,
                                position_ids: Optional[torch.LongTensor] = None,
                                past_key_values: Optional[List[torch.FloatTensor]] = None,
                                inputs_embeds: Optional[torch.FloatTensor] = None,
                                labels: Optional[torch.LongTensor] = None,
                                use_cache: Optional[bool] = None,
                                output_attentions: Optional[bool] = None,
                                output_hidden_states: Optional[bool] = None,
                                return_dict: Optional[bool] = None,
                                stage_manager: Optional[PipelineStageManager] = None,  # this is set by partial
                                hidden_states: Optional[torch.FloatTensor] = None,  # this is from the previous stage
                                ) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        hidden_states=hidden_states,
    )

    hidden_states = outputs[0]
    if not stage_manager.is_last_stage():
        return dict(hidden_states=hidden_states)

    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
    )


class LlamaForCausalLMPolicy(Policy):
    def get_hold_layers(self, module: LlamaForCausalLM) -> List[Module]:
        hold_layers = []

        if self.stage_manager.is_first_stage():
            hold_layers.append(module.model.embed_tokens)
        num_layers_per_stage = len(module.model.layers) // self.stage_manager.num_stages
        hold_layers.extend(module.model.layers[self.stage_manager.stage *
                           num_layers_per_stage: (self.stage_manager.stage + 1) * num_layers_per_stage])
        if self.stage_manager.is_last_stage():
            hold_layers.append(module.model.norm)
            hold_layers.append(module.lm_head)

        return hold_layers

    def get_shared_params(self, module: LlamaForCausalLM) -> List[Dict[int, Tensor]]:
        if id(module.model.embed_tokens.weight) == id(module.lm_head.weight):
            # tie weights
            return [{0: module.model.embed_tokens.weight, self.stage_manager.num_stages - 1: module.lm_head.weight}]
        return []

    def replace_forward(self, module: LlamaForCausalLM) -> None:
        module.model.forward = MethodType(partial(llama_model_forward, stage_manager=self.stage_manager), module.model)
        module.forward = MethodType(partial(llama_for_causal_lm_forward, stage_manager=self.stage_manager), module)
