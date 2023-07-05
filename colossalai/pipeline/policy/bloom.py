import warnings
from functools import partial
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bloom.modeling_bloom import BloomModel
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager

from .base import Policy

logger = logging.get_logger(__name__)


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

    # calculate the num_layers
    num_layers_per_stage = len(self.h) // stage_manager.num_stages
    start_layer = stage_manager.stage * num_layers_per_stage
    end_layer = (stage_manager.stage + 1) * num_layers_per_stage

    for i, (block, layer_past) in enumerate(zip(self.h[start_layer:end_layer], past_key_values[start_layer:end_layer])):
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

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

    # attention_mask is not returned ; presents = past_key_values
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


class BloomModelPolicy(Policy):

    def __init__(self, stage_manager: PipelineStageManager, num_layers: int, num_stages: int):
        super().__init__(stage_manager=stage_manager)
        self.stage_manager = stage_manager
        self.layers_per_stage = self.distribute_layers(num_layers, num_stages)

    def get_hold_layers(self, module: BloomModel) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        hold_layers = []
        if self.stage_manager.is_first_stage():
            hold_layers.append(module.word_embeddings)
            hold_layers.append(module.word_embeddings_layernorm)

        start_idx, end_idx = self.get_stage_index(self.layers_per_stage, self.stage_manager.stage)
        hold_layers.extend(module.h[start_idx:end_idx])

        if self.stage_manager.is_last_stage():
            hold_layers.append(module.ln_f)

        return hold_layers

    def get_shared_params(self, module: BloomModel) -> List[Dict[int, Tensor]]:
        '''no shared params in bloommodel'''
        pass

    def replace_forward(self, module: Module) -> None:
        module.forward = MethodType(partial(bloom_model_forward, stage_manager=self.stage_manager), module.model)
