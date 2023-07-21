from functools import partial
from types import MethodType
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5EncoderModel, T5ForConditionalGeneration, T5Model, T5Stack
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager


class T5PipelineForwards:
    '''
    This class serves as a micro library for forward function substitution of
    T5 models under pipeline setting.
    '''

    @staticmethod
    def t5_stack_forward(
        self: T5Stack,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        encoder_decoder_position_bias: Optional[torch.Tensor] = None,
        stage_index: Optional[List[int]] = None,
        decoder_starting_stage: Optional[int] = None,
    ) -> Union[Dict, Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        # This function is modified on the basis of transformers.models.t5.modeling_t5.T5Stack.forward.
        # Please refer to original code of transformers for more details.

        logger = logging.get_logger(__name__)

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
        if use_cache is True:
            if not in_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        stage = stage_manager.stage
        in_decoder = self.is_decoder
        if in_decoder != (stage >= decoder_starting_stage):
            raise ValueError("Config in T5Stack is not aligned with pipeline setting.")

        # at_first_stage: current stage is the first stage of encoder/decoder, taking input_ids/input_embedds
        # at_last_stage: current stage is the last stage of encoder/decoder, making outputs the same form as huggingface
        at_first_stage = (stage == 0) or (stage == decoder_starting_stage)
        at_last_stage = (stage == decoder_starting_stage - 1) or (stage == stage_manager.num_stages - 1)

        # Process inputs if at the first stage of encoder/decoder.
        if at_first_stage:
            if input_ids is not None and inputs_embeds is not None:
                err_msg_prefix = "decoder_" if in_decoder else ""
                raise ValueError(
                    f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
                )
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                err_msg_prefix = "decoder_" if in_decoder else ""
                raise ValueError(
                    f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")
            if inputs_embeds is None:
                if self.embed_tokens is None:
                    raise ValueError("You have to initialize the model with valid token embeddings")
                inputs_embeds = self.embed_tokens(input_ids)
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
            hidden_states = self.dropout(inputs_embeds)
        else:
            if hidden_states is None:
                raise ValueError(
                    "hidden_states shouldn't be None for stages other than the first stage of encoder/decoder.")
            input_shape = hidden_states.size()[:-1]
            batch_size, seq_length = input_shape[0], input_shape[1]
            device = hidden_states.device

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=device)
        if in_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length, device=device, dtype=torch.long)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        # Going through held blocks.
        start_idx, end_idx = stage_index[0], stage_index[1]

        for i in range(start_idx, end_idx):

            past_key_value = past_key_values[i]
            layer_module = self.block[i]
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            torch.cuda.set_device(hidden_states.device)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,    # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)

            if use_cache is False or use_cache is None:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]
            # print(stage, len(layer_outputs), present_key_value_state.shape)

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]

            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

        # last layer
        if at_last_stage:
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)

            if not return_dict:
                return tuple(v for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ] if v is not None)
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=present_key_value_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )
        else:
            return {
                'hidden_states': hidden_states,
                'position_bias': position_bias,
                'encoder_decoder_position_bias': encoder_decoder_position_bias
            }

    @staticmethod
    def t5_encoder_model_forward(
        self: T5EncoderModel,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        encoder_decoder_position_bias: Optional[torch.Tensor] = None,
        stage_index: Optional[List[int]] = None,
        decoder_starting_stage: Optional[int] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        This function is modified on the basis of transformers.models.t5.modeling_gpt2.T5EncoderModel.forward.
        Please refer to original code of transformers for more details.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = T5PipelineForwards.t5_stack_forward(self.encoder,
                                                      input_ids=input_ids,
                                                      attention_mask=attention_mask,
                                                      inputs_embeds=inputs_embeds,
                                                      head_mask=head_mask,
                                                      output_attentions=output_attentions,
                                                      output_hidden_states=output_hidden_states,
                                                      return_dict=return_dict,
                                                      stage_manager=stage_manager,
                                                      hidden_states=hidden_states,
                                                      position_bias=position_bias,
                                                      encoder_decoder_position_bias=encoder_decoder_position_bias,
                                                      stage_index=stage_index,
                                                      decoder_starting_stage=decoder_starting_stage)

        return outputs
