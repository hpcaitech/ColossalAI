import logging
import random
from functools import partial
from types import MethodType
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.models.opt.modeling_opt import (
    OPTForCausalLM,
    OPTForQuestionAnswering,
    OPTForSequenceClassification,
    OPTModel,
)

from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer import FusedLayerNorm, Linear1D_Col, Linear1D_Row, VocabParallelEmbedding1D

from .._utils import getattr_, setattr_
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    'OPTPolicy', 'OPTModelPolicy', 'OPTForCausalLMPolicy', 'OPTForSequenceClassificationPolicy',
    'OPTForQuestionAnsweringPolicy'
]


class OPTPolicy(Policy):

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
        from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoder, OPTDecoderLayer

        policy = {}

        if self.shard_config.enable_tensor_parallelism:
            policy[OPTDecoder] = ModulePolicyDescription(sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=VocabParallelEmbedding1D,
                )
            ])
            policy[OPTDecoderLayer] = ModulePolicyDescription(sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="fc1",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="fc2",
                    target_module=Linear1D_Row,
                )
            ])

            policy[OPTAttention] = ModulePolicyDescription(attribute_replacement={
                "embed_dim": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size
            },
                                                           sub_module_replacement=[
                                                               SubModuleReplacementDescription(
                                                                   suffix="q_proj",
                                                                   target_module=Linear1D_Col,
                                                               ),
                                                               SubModuleReplacementDescription(
                                                                   suffix="k_proj",
                                                                   target_module=Linear1D_Col,
                                                               ),
                                                               SubModuleReplacementDescription(
                                                                   suffix="v_proj",
                                                                   target_module=Linear1D_Col,
                                                               ),
                                                               SubModuleReplacementDescription(
                                                                   suffix="out_proj",
                                                                   target_module=Linear1D_Row,
                                                               ),
                                                           ])

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="final_layer_norm", target_module=FusedLayerNorm, ignore_if_not_exist=True),
                                                        policy=policy,
                                                        target_key=OPTDecoder)
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(suffix="self_attn_layer_norm",
                                                target_module=FusedLayerNorm,
                                                ignore_if_not_exist=True),
                SubModuleReplacementDescription(suffix="final_layer_norm",
                                                target_module=FusedLayerNorm,
                                                ignore_if_not_exist=True)
            ],
                                                        policy=policy,
                                                        target_key=OPTDecoderLayer)

        return policy

    def postprocess(self):
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == 'OPTModel':
            module = self.model.decoder
        else:
            module = self.model.model.decoder
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.layers), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.embed_tokens)
            held_layers.append(module.embed_positions)
            held_layers.append(module.project_in)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.final_layer_norm)
            held_layers.append(module.project_out)
        return held_layers

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
           to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == 'OPTModel':
                module = self.model.decoder
            else:
                module = self.model.model.decoder

            layers_per_stage = Policy.distribute_layers(len(module.layers), stage_manager.num_stages)
            stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
            method_replacement = {'forward': partial(new_forward, stage_manager=stage_manager, stage_index=stage_index)}
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=model_cls)


class OPTModelPolicy(OPTPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTModel

        policy = super().module_policy()
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(model_cls=OPTModel,
                                      new_forward=OPTPipelineForwards.opt_model_forward,
                                      policy=policy)
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in OPTModel."""
        return []


class OPTForCausalLMPolicy(OPTPolicy):

    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTForCausalLM

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="lm_head", target_module=Linear1D_Col, kwargs=dict(gather_output=True)),
                                                        policy=policy,
                                                        target_key=OPTForCausalLM)

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(model_cls=OPTForCausalLM,
                                      new_forward=OPTPipelineForwards.opt_for_causal_lm_forward,
                                      policy=policy)

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        opt_model = self.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            num_stages = self.pipeline_stage_manager.num_stages
            if id(opt_model.model.decoder.embed_tokens.weight) == id(opt_model.lm_head.weight):
                return [{0: opt_model.model.decoder.embed_tokens.weight, num_stages - 1: opt_model.lm_head.weight}]

    def postprocess(self):
        if self.shard_config.enable_tensor_parallelism and self.pipeline_stage_manager is None:
            binding_map = {
                'model.decoder.embed_tokens': 'lm_head',
            }

            for k, v in binding_map.items():
                src_mod = getattr_(self.model, k)
                dst_mod = getattr_(self.model, v)
                dst_mod.weight = src_mod.weight

        return self.model


class OPTForSequenceClassificationPolicy(OPTPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTForSequenceClassification

        policy = super().module_policy()
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(model_cls=OPTForSequenceClassification,
                                      new_forward=OPTPipelineForwards.opt_for_sequence_classification_forward,
                                      policy=policy)

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.score)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        "no shared params in OPTForSequenceClassification"
        return []


class OPTForQuestionAnsweringPolicy(OPTPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTForQuestionAnswering

        policy = super().module_policy()
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(model_cls=OPTForQuestionAnswering,
                                      new_forward=OPTPipelineForwards.opt_for_question_answering_forward,
                                      policy=policy)

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.qa_outputs)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        "no shared params in OPTForSequenceClassification"
        return []


class OPTPipelineForwards:
    '''
    This class serves as a micro library for forward function substitution of OPT models
    under pipeline setting.
    '''

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, _dtype, device, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        from transformers.models.opt.modeling_opt import _make_causal_mask
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                _dtype,
                device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = OPTPipelineForwards._expand_mask(attention_mask, _dtype,
                                                                  tgt_len=input_shape[-1]).to(device)
            combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                                       combined_attention_mask)

        return combined_attention_mask

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    @staticmethod
    def opt_model_forward(
        self: OPTModel,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        '''
        This forward method is modified based on transformers.models.opt.modeling_opt.OPTModel.forward
        '''

        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.utils import logging
        logger = logging.get_logger(__name__)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        decoder = self.decoder
        if stage_manager.is_first_stage():
            # retrieve input_ids and inputs_embeds
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

            batch_size, seq_length = input_shape

            if inputs_embeds is None:
                inputs_embeds = decoder.embed_tokens(input_ids)

            if decoder.project_in is not None:
                inputs_embeds = decoder.project_in(inputs_embeds)
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            _dtype = inputs_embeds.dtype

        else:
            if hidden_states is None:
                raise ValueError("hidden_states shouln't be None for intermediate stages.")
            input_shape = hidden_states.size()[:-1]
            batch_size, seq_length = input_shape[0], input_shape[1]
            device = hidden_states.device
            _dtype = hidden_states.dtype

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=device)
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)")

        causal_attention_mask = OPTPipelineForwards._prepare_decoder_attention_mask(attention_mask, input_shape, _dtype,
                                                                                    device, past_key_values_length)

        if stage_manager.is_first_stage():
            pos_embeds = decoder.embed_positions(attention_mask, past_key_values_length)
            hidden_states = inputs_embeds + pos_embeds

        if decoder.gradient_checkpointing and decoder.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        # TODO: left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if past_key_values:
            logger.warning_once('Non-empty past_key_values is not supported for pipeline models at the moment.')
            past_key_values = None
        if output_attentions:
            logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
            output_attentions = False
        if output_hidden_states:
            logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
            output_hidden_states = False
        if use_cache:
            logger.warning_once('use_cache=True is not supported for pipeline models at the moment.')
            use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(decoder.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(decoder.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}.")

        start_idx, end_idx = stage_index[0], stage_index[1]

        torch.cuda.set_device(device)

        for idx in range(start_idx, end_idx):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            decoder_layer = decoder.layers[idx]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            dropout_probability = random.uniform(0, 1)
            if decoder.training and (dropout_probability < decoder.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if decoder.gradient_checkpointing and decoder.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
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
            if decoder.final_layer_norm is not None:
                hidden_states = decoder.final_layer_norm(hidden_states)
            if decoder.project_out is not None:
                hidden_states = decoder.project_out(hidden_states)

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
        else:
            return {'hidden_states': hidden_states}

    @staticmethod
    def opt_for_causal_lm_forward(
        self: OPTForCausalLM,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        from transformers.modeling_outputs import CausalLMOutputWithPast

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = OPTPipelineForwards.opt_model_forward(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
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
        if stage_manager.is_last_stage():
            logits = self.lm_head(outputs[0]).contiguous()
            loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(logits.device)
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
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

    @staticmethod
    def opt_for_sequence_classification_forward(
        self: OPTForSequenceClassification,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        from transformers.modeling_outputs import SequenceClassifierOutputWithPast
        from transformers.utils import logging
        logger = logging.get_logger(__name__)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = OPTPipelineForwards.opt_model_forward(self.model,
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

        if stage_manager.is_last_stage():
            hidden_states = transformer_outputs[0]
            logits = self.score(hidden_states)

            batch_size = input_ids.shape[0] if input_ids is not None else hidden_states.shape[0]

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

    @staticmethod
    def opt_for_question_answering_forward(
        self: OPTForQuestionAnswering,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForQuestionAnswering
        >>> import torch

        >>> torch.manual_seed(4)  # doctest: +IGNORE_RESULT
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> # note: we are loading a OPTForQuestionAnswering from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = OPTForQuestionAnswering.from_pretrained("facebook/opt-350m")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

        >>> inputs = tokenizer(question, text, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> answer_offset = len(tokenizer(question)[0])

        >>> predict_answer_tokens = inputs.input_ids[
        ...     0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
        ... ]
        >>> predicted = tokenizer.decode(predict_answer_tokens)
        >>> predicted
        ' a nice puppet'
        ```"""
        from transformers.modeling_outputs import QuestionAnsweringModelOutput

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = OPTPipelineForwards.opt_model_forward(self.model,
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
        if stage_manager.is_last_stage():
            hidden_states = transformer_outputs[0]

            logits = self.qa_outputs(hidden_states)
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
                output = (start_logits, end_logits) + transformer_outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output

            return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        else:
            hidden_states = transformer_outputs.get('hidden_states')
            return {'hidden_states': hidden_states}
