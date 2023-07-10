import math
from functools import partial
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.utils import ModelOutput, logging

from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer import FusedRMSNorm, Linear1D_Col, Linear1D_Row, VocabParallelEmbedding1D

from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

logger = logging.get_logger(__name__)

__all__ = ['LlamaPolicy', 'LlamaForCausalLMPolicy', 'LlamaForSequenceClassificationPolicy']


class LlamaPolicy(Policy):

    def config_sanity_check(self):
        pass

    def preprocess(self):
        if self.shard_config.enable_tensor_parallelism:
            # Resize embedding
            vocab_size = self.model.config.vocab_size
            world_size = self.shard_config.tensor_parallel_size

            if vocab_size % world_size != 0:
                new_vocab_size = vocab_size + world_size - vocab_size % world_size
                self.model.resize_token_embeddings(new_vocab_size)

        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel

        policy = {}

        if self.shard_config.enable_tensor_parallelism:
            policy[LlamaDecoderLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attn.hidden_size":
                        self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                    "self_attn.num_heads":
                        self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=Linear1D_Row,
                    )
                ],
            )

            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="embed_tokens",
                target_module=VocabParallelEmbedding1D,
            ),
                                                        policy=policy,
                                                        target_key=LlamaModel)

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(
                    suffix="input_layernorm",
                    target_module=FusedRMSNorm,
                ),
                SubModuleReplacementDescription(
                    suffix="post_attention_layernorm",
                    target_module=FusedRMSNorm,
                )
            ],
                                                        policy=policy,
                                                        target_key=LlamaDecoderLayer)

            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="norm",
                target_module=FusedRMSNorm,
            ),
                                                        policy=policy,
                                                        target_key=LlamaModel)

        return policy

    def postprocess(self):
        return self.model


class LlamaModelPolicy(LlamaPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        module_policy = super().module_policy()
        from transformers.models.llama.modeling_llama import LlamaModel
        if self.pipeline_stage_manager:
            # set None as default
            module_policy[LlamaModel] = ModulePolicyDescription(
                method_replacement={'forward': partial(llama_model_forward, stage_manager=self.pipeline_stage_manager)})
        return module_policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.layers), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.embed_tokens)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.norm)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in bert model"""
        return []


class LlamaForCausalLMPolicy(LlamaPolicy):

    def module_policy(self):
        from transformers import LlamaForCausalLM

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for casual lm
            new_item = {
                LlamaForCausalLM:
                    ModulePolicyDescription(sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head", target_module=Linear1D_Col, kwargs=dict(gather_output=True))
                    ])
            }
            policy.update(new_item)
        return policy


class LlamaForSequenceClassificationPolicy(LlamaPolicy):

    def module_policy(self):
        from transformers import LlamaForSequenceClassification

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for sequence classification
            new_item = {
                LlamaForSequenceClassification:
                    ModulePolicyDescription(sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="score", target_module=Linear1D_Col, kwargs=dict(gather_output=True))
                    ])
            }
            policy.update(new_item)
        return policy


def llama_model_forward(
    self: LlamaModel,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    stage_manager: Optional[PipelineStageManager] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    stage_index: Optional[List[int]] = None,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (output_hidden_states
                            if output_hidden_states is not None else self.config.output_hidden_states)
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if stage_manager.is_first_stage():
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
    else:
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_length = input_shape
        device = hidden_states.device

    seq_length_with_past = seq_length
    past_key_values_length = 0

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

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        position_ids = torch.arange(past_key_values_length,
                                    seq_length + past_key_values_length,
                                    dtype=torch.long,
                                    device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    # embed positions, for the first stage, hidden_states is the input embeddings,
    # for the other stages, hidden_states is the output of the previous stage
    # TODO: we should recive the attn mask of 1st stage and send it to the other stages
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
    attention_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), hidden_states,
                                                          past_key_values_length)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    start_idx, end_idx = stage_index[0], stage_index[1]
    for idx, decoder_layer in enumerate(self.layers[start_idx:end_idx]):
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
    if stage_manager.is_last_stage():
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    # always return dict for imediate stage
    return {'hidden_states': hidden_states, 'attention_mask': attention_mask}
