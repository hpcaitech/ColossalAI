"""Whisper adapters for Transformers 5.17 Cache and Tensor layer contracts."""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)
from transformers.models.whisper.modeling_whisper import (
    _HIDDEN_STATES_START_POSITION,
    WhisperForAudioClassification,
    WhisperForConditionalGeneration,
    WhisperModel,
    shift_tokens_right,
)
from transformers.utils import logging
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer import ColoAttention
from colossalai.shardformer.shard import ShardConfig

from .._utils import (
    get_hf_cache_seq_length,
    get_hf_flash_attention_kwargs,
    get_hf_mask_config,
    prepare_hf_encoder_decoder_cache,
)

logger = logging.get_logger(__name__)


def _get_attention_mask(
    self,
    shard_config,
    hidden_states,
    past_key_values_length,
    attention_mask,
    head_mask=None,
    output_attentions=False,
    *,
    past_key_values=None,
    position_ids=None,
):
    if head_mask is not None:
        raise ValueError("HF 5.17 Whisper does not support head_mask")
    flash = shard_config is not None and shard_config.enable_flash_attention
    config = get_hf_mask_config(self.config, eager=flash)
    if past_key_values is None and past_key_values_length:
        # Preserve the length-only helper interface for callers without a Cache.
        # Normal decode below always supplies the real Cache to masking_utils.
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        return _prepare_4d_causal_attention_mask(
            attention_mask, hidden_states.shape[:2], hidden_states, past_key_values_length
        )
    return create_causal_mask(
        config=config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )


def get_whisper_flash_attention_forward():
    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_values=None,
        attention_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        is_cross = key_value_states is not None
        input_shape = hidden_states.shape[:-1]
        # Match HF's order of scaling (Whisper is sensitive to moving this
        # multiplication after the QK product).
        query = (
            (self.q_proj(hidden_states) * self.scaling)
            .view(*input_shape, -1, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        cache = past_key_values
        updated = False
        if isinstance(cache, EncoderDecoderCache):
            updated = cache.is_updated.get(self.layer_idx, False)
            cache = cache.cross_attention_cache if is_cross else cache.self_attention_cache
        if is_cross and cache is not None and updated:
            key = cache.layers[self.layer_idx].keys
            value = cache.layers[self.layer_idx].values
        else:
            states = key_value_states if is_cross else hidden_states
            shape = (input_shape[0], -1, self.num_heads, self.head_dim)
            key = self.k_proj(states).view(shape).transpose(1, 2).contiguous()
            value = self.v_proj(states).view(shape).transpose(1, 2).contiguous()
            if cache is not None:
                key, value = cache.update(key, value, self.layer_idx)
                if is_cross and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True
        output = ColoAttention.attention(
            query,
            key,
            value,
            **get_hf_flash_attention_kwargs(attention_mask),
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0,
        )
        output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.out_proj(output), None

    return forward


def get_whisper_decoder_forward_for_flash_attention(shard_config):
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("Whisper checkpointing disables use_cache.")
            use_cache = False
            past_key_values = None
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache:
            past_key_values = prepare_hf_encoder_decoder_cache(
                self.config, past_key_values, encoder_hidden_states=encoder_hidden_states
            )
        past_length = get_hf_cache_seq_length(past_key_values)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_length
            position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)
        positions = self.embed_positions(inputs_embeds, past_key_values_length=past_length, position_ids=position_ids)
        hidden_states = nn.functional.dropout(
            inputs_embeds + positions.to(inputs_embeds.device), p=self.dropout, training=self.training
        )
        mask = _get_attention_mask(
            self,
            shard_config,
            inputs_embeds,
            past_length,
            attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        for layer in self.layers:
            if self.training and torch.rand([]) < self.layerdrop:
                continue
            # Encoder states need positional binding for reentrant checkpoint
            # to track the cross-attention gradient back into the encoder.
            hidden_states = layer(
                hidden_states,
                mask,
                encoder_hidden_states,
                encoder_attention_mask=None,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
                **kwargs,
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=self.layer_norm(hidden_states),
            past_key_values=past_key_values,
        )

    return forward


def get_jit_fused_whisper_encoder_layer_forward():
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            self.self_attn_layer_norm(hidden_states), attention_mask=attention_mask, **kwargs
        )
        hidden_states = self.dropout_add(hidden_states, residual, self.dropout, self.training)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(self.final_layer_norm(hidden_states)))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.dropout_add(self.fc2(hidden_states), residual, self.dropout, self.training)
        if hidden_states.dtype == torch.float16:
            bound = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = hidden_states.clamp(-bound, bound)
        return hidden_states

    return forward


def get_jit_fused_whisper_decoder_layer_forward():
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=True,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            self.self_attn_layer_norm(hidden_states),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.dropout_add(hidden_states, residual, self.dropout, self.training)
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states, _ = self.encoder_attn(
                self.encoder_attn_layer_norm(hidden_states),
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            hidden_states = self.dropout_add(hidden_states, residual, self.dropout, self.training)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(self.final_layer_norm(hidden_states)))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        return self.dropout_add(self.fc2(hidden_states), residual, self.dropout, self.training)

    return forward


class WhisperPipelineForwards:
    """Pipeline stage functions; Cache generation remains a non-pipeline path."""

    @staticmethod
    def whisper_encoder_forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        stage_manager=None,
        hidden_states=None,
        encoder_states=None,
        all_attentions=None,
        stage_index=None,
        decoder_starting_stage=None,
        shard_config=None,
    ):
        if head_mask is not None:
            raise ValueError("HF 5.17 Whisper does not support head_mask")
        if output_attentions:
            logger.warning_once("Pipeline Whisper does not collect attention weights.")
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        if stage_manager.stage == 0:
            expected = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
            if input_features.shape[-1] != expected:
                raise ValueError(f"Whisper expects mel input length {expected}, got {input_features.shape[-1]}")
            hidden_states = nn.functional.gelu(self.conv1(input_features))
            hidden_states = nn.functional.gelu(self.conv2(hidden_states)).permute(0, 2, 1)
            positions = torch.arange(self.embed_positions.num_embeddings, device=hidden_states.device)
            hidden_states = nn.functional.dropout(
                hidden_states + self.embed_positions(positions), p=self.dropout, training=self.training
            )
            encoder_states = () if output_hidden_states else None
        elif hidden_states is None:
            raise ValueError("hidden_states is required at non-first encoder stages")
        if output_hidden_states and encoder_states is None:
            raise ValueError("encoder_states must be forwarded when collecting pipeline hidden states")
        for idx in range(*stage_index):
            if output_hidden_states:
                encoder_states += (hidden_states,)
            if self.training and torch.rand([]) < self.layerdrop:
                continue
            hidden_states = self.layers[idx](hidden_states, attention_mask=None)
        if stage_manager.stage == decoder_starting_stage - 1:
            hidden_states = self.layer_norm(hidden_states)
            if output_hidden_states:
                encoder_states += (hidden_states,)
            outputs = BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)
            return outputs if return_dict else outputs.to_tuple()
        outputs = {"hidden_states": hidden_states}
        if output_hidden_states:
            outputs["encoder_states"] = encoder_states
        return outputs

    @staticmethod
    def whisper_decoder_forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        stage_manager=None,
        hidden_states=None,
        stage_index=None,
        decoder_starting_stage=None,
        shard_config=None,
    ):
        if head_mask is not None or cross_attn_head_mask is not None:
            raise ValueError("HF 5.17 Whisper does not support head masks")
        if past_key_values is not None or use_cache:
            logger.warning_once("Pipeline Whisper does not support Cache generation; cache is disabled.")
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if stage_manager.stage == decoder_starting_stage:
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("Specify exactly one of input_ids and inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            positions = self.embed_positions(inputs_embeds, position_ids=position_ids)
            hidden_states = nn.functional.dropout(
                inputs_embeds + positions.to(inputs_embeds.device), p=self.dropout, training=self.training
            )
        elif hidden_states is None:
            raise ValueError("hidden_states is required at non-first decoder stages")
        mask = _get_attention_mask(self, shard_config, hidden_states, 0, attention_mask, position_ids=position_ids)
        for idx in range(*stage_index):
            if self.training and torch.rand([]) < self.layerdrop:
                continue
            hidden_states = self.layers[idx](
                hidden_states,
                mask,
                encoder_hidden_states,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=False,
            )
        if not stage_manager.is_last_stage():
            return {"hidden_states": hidden_states}
        outputs = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=self.layer_norm(hidden_states))
        return outputs if return_dict else outputs.to_tuple()

    @staticmethod
    def whisper_model_forward(
        self: WhisperModel,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Cache] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        decoder_starting_stage: Optional[int] = None,
        shard_config: Optional[ShardConfig] = None,
    ):
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import AutoFeatureExtractor, WhisperModel
         >>> from datasets import load_dataset

         >>> model = WhisperModel.from_pretrained("openai/whisper-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```"""
        # Pipeline generation is not supported; never re-enable cache from config defaults.
        if past_key_values is not None:
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

        use_cache = False
        output_attentions = False
        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        in_decoder = stage_manager.stage >= decoder_starting_stage
        if not in_decoder:
            if encoder_outputs is None:
                if stage_manager.stage == 0:
                    input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

                encoder_outputs = WhisperPipelineForwards.whisper_encoder_forward(
                    self.encoder,
                    input_features,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    stage_manager=stage_manager,
                    hidden_states=hidden_states,
                    stage_index=stage_index,
                    decoder_starting_stage=decoder_starting_stage,
                )

                if stage_manager.stage == decoder_starting_stage - 1:
                    # last stage of encoder
                    return {"encoder_hidden_states": encoder_outputs[0]}
                else:
                    return encoder_outputs
            # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=(encoder_outputs[1] if len(encoder_outputs) > 1 else None),
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                )

        at_last_decoder_stage = stage_manager.is_last_stage()
        at_first_decoder_stage = stage_manager.stage == decoder_starting_stage
        if encoder_outputs is not None:
            encoder_hidden_states = encoder_outputs[0]
        elif encoder_hidden_states is None:
            raise ValueError("Non-empty encoder_hidden_states should be passed in at decoder stages.")

        if not at_first_decoder_stage and hidden_states is None:
            raise ValueError("If not at the first layer of decoder, non-empty hidden_states must be provided.")

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = WhisperPipelineForwards.whisper_decoder_forward(
            self.decoder,
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            decoder_starting_stage=decoder_starting_stage,
            shard_config=shard_config,
        )

        # Directly return outputs of overloaded Whisper forward if not at last stage.
        if not at_last_decoder_stage:
            # encoder_hidden_states should be passed to the next stage
            decoder_outputs["encoder_hidden_states"] = encoder_hidden_states
            return decoder_outputs

        if not return_dict:
            return decoder_outputs + (encoder_hidden_states,)

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
        )

    @staticmethod
    def whisper_for_conditional_generation_forward(
        self: WhisperForConditionalGeneration,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Cache] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        decoder_starting_stage: Optional[int] = None,
        shard_config: Optional[ShardConfig] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        in_decoder = stage_manager.stage >= decoder_starting_stage
        at_last_decoder_stage = stage_manager.is_last_stage()
        outputs = WhisperPipelineForwards.whisper_model_forward(
            self.model,
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            stage_index=stage_index,
            decoder_starting_stage=decoder_starting_stage,
            shard_config=shard_config,
        )
        if not in_decoder:
            return outputs

        if not at_last_decoder_stage:
            # encoder_hidden_states should be passed to the next stage
            outputs["encoder_hidden_states"] = encoder_hidden_states
            return outputs

        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @staticmethod
    def whisper_for_audio_classification_forward(
        self: WhisperForAudioClassification,
        input_features: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        encoder_states=None,
        all_attentions=None,
        stage_index: Optional[List[int]] = None,
        decoder_starting_stage: Optional[int] = None,
        shard_config: Optional[ShardConfig] = None,
    ):
        r"""
        This function is modified on the basis of transformers.models.whisper.modeling_whisper.WhisperForAudioClassification.forward.
        Please refer to original code of transformers for more details.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if self.config.use_weighted_layer_sum:
            output_hidden_states = True
        elif output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # audio_classification only holds encoder
        encoder_outputs = WhisperPipelineForwards.whisper_encoder_forward(
            self.encoder,
            input_features,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            encoder_states=encoder_states,
            stage_index=stage_index,
            decoder_starting_stage=decoder_starting_stage,
        )

        if not stage_manager.is_last_stage():
            return encoder_outputs

        if self.config.use_weighted_layer_sum:
            hidden_states = encoder_outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = encoder_outputs[0]

        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)

        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
