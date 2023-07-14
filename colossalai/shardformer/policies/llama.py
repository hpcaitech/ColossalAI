from functools import partial
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel
from transformers.utils import logging

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
        policy = super().module_policy()
        from transformers.models.llama.modeling_llama import LlamaModel
        if self.pipeline_stage_manager:
            # set None as default
            stage_manager = self.pipeline_stage_manager
            layers_per_stage = Policy.distribute_layers(len(self.model.layers), stage_manager.num_stages)
            stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
            method_replacement = {
                'forward': partial(llama_model_forward, stage_manager=stage_manager, stage_index=stage_index)
            }
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=LlamaModel)
        return policy

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
        """No shared params in llama model"""
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

        if self.pipeline_stage_manager:
            # set None as default
            stage_manager = self.pipeline_stage_manager
            layers_per_stage = Policy.distribute_layers(len(self.model.model.layers), stage_manager.num_stages)
            stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
            method_replacement = {
                'forward': partial(llama_for_causal_lm_forward, stage_manager=stage_manager, stage_index=stage_index)
            }
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=LlamaForCausalLM)
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.model.layers), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.model.embed_tokens)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.model.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.model.norm)
            held_layers.append(module.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        llama_model = self.model.model
        if id(llama_model.embed_tokens.weight) == id(self.model.lm_head.weight):
            # tie weights
            return [{
                0: llama_model.embed_tokens.weight,
                self.pipeline_stage_manager.num_stages - 1: self.model.lm_head.weight
            }]
        return []


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
        # to be confirmed
        if self.pipeline_stage_manager:
            # set None as default
            stage_manager = self.pipeline_stage_manager
            layers_per_stage = Policy.distribute_layers(len(self.model.model.layers), stage_manager.num_stages)
            stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
            method_replacement = {
                'forward':
                    partial(llama_for_sequence_classification_forward,
                            stage_manager=stage_manager,
                            stage_index=stage_index)
            }
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=LlamaForSequenceClassification)
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.model.layers), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.model.embed_tokens)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.model.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.model.norm)
            held_layers.append(module.score)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in llama for sequence classification model"""
        return []


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
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device)
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
    return {'hidden_states': hidden_states}


def llama_for_causal_lm_forward(
    self: LlamaForCausalLM,
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
    stage_manager: Optional[PipelineStageManager] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    stage_index: Optional[List[int]] = None,
):
    r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (output_hidden_states
                            if output_hidden_states is not None else self.config.output_hidden_states)
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

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = llama_model_forward(
        self.model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
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
        hidden_states = outputs[0]
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
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    else:
        hidden_states = outputs.get('hidden_states')
        return {'hidden_states': hidden_states}


def llama_for_sequence_classification_forward(
    self: LlamaForSequenceClassification,
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
    stage_manager: Optional[PipelineStageManager] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    stage_index: Optional[List[int]] = None,
):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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

    transformer_outputs = llama_model_forward(
        self.model,
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        stage_manager=stage_manager,
        hidden_states=hidden_states,
        stage_index=stage_index,
    )

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        batch_size = inputs_embeds.shape[0]
    else:
        batch_size = hidden_states.shape[0]

    if stage_manager.is_last_stage():
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

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
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
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
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
