import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    TokenClassifierOutput,
)
from transformers.models.t5.modeling_t5 import (
    T5EncoderModel,
    T5ForConditionalGeneration,
    T5ForTokenClassification,
    T5Model,
    T5Stack,
)
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager


class T5PipelineForwards:
    """
    This class serves as a micro library for forward function substitution of
    T5 models under pipeline setting.
    """

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

        # TODO(baizhou): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if past_key_values:
            logger.warning_once("Non-empty past_key_values is not supported for pipeline models at the moment.")
            past_key_values = None
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False
        if use_cache is True:
            if not in_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        stage = stage_manager.stage
        in_decoder = self.is_decoder
        if in_decoder != (stage >= decoder_starting_stage):
            raise ValueError("Config in T5Stack is not aligned with pipeline setting.")

        # at_first_stage: current stage is the first stage of encoder/decoder, taking input_ids/input_embeds
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
                    f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
                )
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
                    "hidden_states shouldn't be None for stages other than the first stage of encoder/decoder."
                )
            input_shape = hidden_states.size()[:-1]
            batch_size, seq_length = input_shape[0], input_shape[1]
            device = hidden_states.device

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long)
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
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
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

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]

            if in_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

        # last layer
        if at_last_stage:
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)

            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        present_key_value_states,
                        all_hidden_states,
                        all_attentions,
                        all_cross_attentions,
                    ]
                    if v is not None
                )
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=present_key_value_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )
        else:
            return {
                "hidden_states": hidden_states,
                "position_bias": position_bias,
                "encoder_decoder_position_bias": encoder_decoder_position_bias,
                "backward_tensor_keys": ["hidden_states"],
            }

    @staticmethod
    def t5_model_forward(
        self: T5Model,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        encoder_decoder_position_bias: Optional[torch.Tensor] = None,
        backward_tensor_keys: Optional[List[str]] = None,
        stage_index: Optional[List[int]] = None,
        decoder_starting_stage: Optional[int] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        # This function is modified on the basis of transformers.models.t5.modeling_t5.T5Model.forward.
        # Please refer to original code of transformers for more details.

        __HEAD_MASK_WARNING_MSG = """
        The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
        `decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
        If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
        num_heads)`.
        """

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logger = logging.get_logger(__name__)

        # TODO(baizhou): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if past_key_values:
            logger.warning_once("Non-empty past_key_values is not supported for pipeline models at the moment.")
            past_key_values = None
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        in_decoder = stage_manager.stage >= decoder_starting_stage
        # Stage is in encoder, directly return the output of t5_stack_forward
        if not in_decoder:
            encoder_outputs = T5PipelineForwards.t5_stack_forward(
                self.encoder,
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
                decoder_starting_stage=decoder_starting_stage,
            )
            if stage_manager.stage == decoder_starting_stage - 1:
                # last stage of encoder
                return {"encoder_hidden_states": encoder_outputs[0]}
            else:
                return encoder_outputs

        at_last_decoder_stage = stage_manager.is_last_stage()
        at_first_decoder_stage = stage_manager.stage == decoder_starting_stage

        if encoder_outputs is not None:
            encoder_hidden_states = encoder_outputs[0]
        elif encoder_hidden_states is None:
            raise ValueError("Non-empty encoder_hidden_states should be passed in at decoder stages.")

        if not at_first_decoder_stage and hidden_states is None:
            raise ValueError("If not at the first layer of decoder, non-empty hidden_states must be provided.")

        # Decode
        decoder_outputs = T5PipelineForwards.t5_stack_forward(
            self.decoder,
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            position_bias=position_bias,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            stage_index=stage_index,
            decoder_starting_stage=decoder_starting_stage,
        )

        # Directly return outputs of overloaded T5Stack forward if not at last stage.
        if not at_last_decoder_stage:
            # encoder_hidden_states should be passed to the next stage
            decoder_outputs["encoder_hidden_states"] = encoder_hidden_states
            return decoder_outputs

        if not return_dict:
            return decoder_outputs + encoder_hidden_states
        else:
            return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_hidden_states,
            )

    @staticmethod
    def t5_for_conditional_generation_forward(
        self: T5ForConditionalGeneration,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        encoder_decoder_position_bias: Optional[torch.Tensor] = None,
        backward_tensor_keys: Optional[List[str]] = None,
        stage_index: Optional[List[int]] = None,
        decoder_starting_stage: Optional[int] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        # This function is modified on the basis of transformers.models.t5.modeling_t5.T5ForConditionalGeneration.forward.
        # Please refer to original code of transformers for more details.

        __HEAD_MASK_WARNING_MSG = """
        The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
        `decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
        If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
        num_heads)`.
        """

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logger = logging.get_logger(__name__)

        # TODO(baizhou): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if past_key_values:
            logger.warning_once("Non-empty past_key_values is not supported for pipeline models at the moment.")
            past_key_values = None
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        in_decoder = stage_manager.stage >= decoder_starting_stage

        # Stage is in encoder, directly return the output of t5_stack_forward
        if not in_decoder:
            encoder_outputs = T5PipelineForwards.t5_stack_forward(
                self.encoder,
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
                decoder_starting_stage=decoder_starting_stage,
            )
            if stage_manager.stage == decoder_starting_stage - 1:
                # last stage of encoder
                return {"encoder_hidden_states": encoder_outputs[0]}
            else:
                return encoder_outputs

        at_last_decoder_stage = stage_manager.is_last_stage()
        at_first_decoder_stage = stage_manager.stage == decoder_starting_stage

        if encoder_outputs is not None:
            encoder_hidden_states = encoder_outputs[0]
        elif encoder_hidden_states is None:
            raise ValueError("Non-empty encoder_hidden_states should be passed in at decoder stages.")

        if not at_first_decoder_stage and hidden_states is None:
            raise ValueError("If not at the first layer of decoder, non-empty hidden_states must be provided.")

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = T5PipelineForwards.t5_stack_forward(
            self.decoder,
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            position_bias=position_bias,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            stage_index=stage_index,
            decoder_starting_stage=decoder_starting_stage,
        )

        # Directly return outputs of overloaded T5Stack forward if not at last stage.
        if not at_last_decoder_stage:
            # encoder_hidden_states should be passed to the next stage
            decoder_outputs["encoder_hidden_states"] = encoder_hidden_states
            return decoder_outputs

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_hidden_states
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
        )

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
        backward_tensor_keys: Optional[List[str]] = None,
        stage_index: Optional[List[int]] = None,
        decoder_starting_stage: Optional[int] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        This function is modified on the basis of transformers.models.t5.modeling_gpt2.T5EncoderModel.forward.
        Please refer to original code of transformers for more details.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = T5PipelineForwards.t5_stack_forward(
            self.encoder,
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
            decoder_starting_stage=decoder_starting_stage,
        )

        return outputs

    @staticmethod
    def t5_for_token_classification_forward(
        self: T5ForTokenClassification,
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
        labels: Optional[torch.LongTensor] = None,
        backward_tensor_keys: Optional[List[str]] = None,
        stage_index: Optional[List[int]] = None,
        decoder_starting_stage: Optional[int] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        This function is modified on the basis of transformers.models.t5.modeling_t5.T5ForTokenClassification.forward.
        Please refer to original code of transformers for more details.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = T5PipelineForwards.t5_stack_forward(
            self.transformer.encoder,
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
            decoder_starting_stage=decoder_starting_stage,
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

        return outputs


def get_t5_flash_attention_forward():
    from transformers.models.t5.modeling_t5 import T5Attention

    def forward(
        self: T5Attention,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        query_length: Optional[int] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=query_states.device, dtype=query_states.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=query_states.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=position_bias_masked,
                dropout_p=self.dropout,
                scale=1.0,
            )
        attn_output = unshape(attn_output)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None

        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        return outputs

    return forward


def get_jit_fused_T5_layer_ff_forward():
    from transformers.models.t5.modeling_t5 import T5LayerFF

    def forward(self: T5LayerFF, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = self.dropout_add(forwarded_states, hidden_states, self.dropout.p, self.dropout.training)
        return hidden_states

    return forward


def get_T5_layer_self_attention_forward():
    from transformers.models.t5.modeling_t5 import T5LayerSelfAttention

    def forward(
        self: T5LayerSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout_add(attention_output[0], hidden_states, self.dropout.p, self.dropout.training)
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

    return forward


def get_T5_layer_cross_attention_forward():
    from transformers.models.t5.modeling_t5 import T5LayerCrossAttention

    def forward(
        self: T5LayerCrossAttention,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        query_length: Optional[int] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = self.dropout_add(attention_output[0], hidden_states, self.dropout.p, self.dropout.training)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

    return forward
