""" PyTorch ChatGLM model. """

import copy
import math
import re
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn.utils import skip_init
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import GenerationConfig, LogitsProcessorList, ModelOutput, StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

import colossalai
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.tests.kit.model_zoo.transformers.chatglm2_6b.configuration_chatglm import ChatGLMConfig
from colossalai.tests.kit.model_zoo.transformers.chatglm2_6b.modeling_chatglm import ChatGLMModel, GLMBlock

logger = logging.get_logger(__name__)


def chatglm_model_forward(
    self,
    input_ids,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.BoolTensor] = None,
    full_attention_mask: Optional[torch.BoolTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    stage_index: Optional[List[int]] = None,
):

    output_hidden_states = (output_hidden_states
                            if output_hidden_states is not None else self.config.output_hidden_states)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if stage_manager.is_first_stage():
        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        hidden_states = inputs_embeds
    else:
        batch_size, seq_length = hidden_states.shape[:2]

    if self.pre_seq_len is not None:
        if past_key_values is None:
            past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device, dtype=inputs_embeds.dtype)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)), attention_mask],
                                       dim=-1)

    if full_attention_mask is None:
        if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
            full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

    # Rotary positional embeddings
    rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
    if position_ids is not None:
        rotary_pos_emb = rotary_pos_emb[position_ids]
    else:
        rotary_pos_emb = rotary_pos_emb[None, :seq_length]
    rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

    if not kv_caches:
        kv_caches = [None for _ in range(self.num_layers)]

    presents = () if use_cache else None

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

    all_self_attentions = None
    all_hidden_states = () if output_hidden_states else None
    start_idx, end_idx = stage_index[0], stage_index[1]
    for idx in range(start_idx, end_idx):
        layer = self.encoder._get_layer(idx)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_ret = torch.utils.checkpoint.checkpoint(layer, hidden_states, attention_mask, rotary_pos_emb,
                                                          kv_caches[idx], use_cache)
        else:
            layer_ret = layer(hidden_states,
                              attention_mask,
                              rotary_pos_emb,
                              kv_cache=kv_caches[idx],
                              use_cache=use_cache)
        hidden_states, kv_cache = layer_ret
        if use_cache:
            presents = presents + (kv_cache,)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if stage_manager.is_last_stage():
        # final layer_norm
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    else:
        return {'hidden_states': hidden_states}
