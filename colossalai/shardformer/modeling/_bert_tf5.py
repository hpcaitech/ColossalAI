import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.models.bert.modeling_bert import (
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForPreTrainingOutput,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertLMHeadModel,
    BertModel,
)
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig
from colossalai.shardformer.layer._operation import gather_forward_split_backward, split_forward_gather_backward


class BertPipelineForwards:
    """
    This class serves as a micro library for forward function substitution of Bert models
    under pipeline setting.
    """

    @staticmethod
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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,  # this is from the previous stage
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        # TODO(jianghai): add explaination of the output here.
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
        logger = logging.get_logger(__name__)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
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
            if token_type_ids is None:
                if hasattr(self.embeddings, "token_type_ids"):
                    buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        else:
            input_shape = hidden_states.size()[:-1]
            batch_size, seq_length = input_shape
            device = hidden_states.device

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False

        # `head_mask` is gone in v5 (`get_head_mask` was removed too). The other PP forwards still
        # pass it by name, so the parameter stays; a non-None value warns instead of being
        # swallowed by v5's `**kwargs` like upstream does.
        if head_mask is not None:
            logger.warning_once(
                "`head_mask` is not supported for pipeline models at the moment and was removed in "
                "transformers v5; the value is ignored."
            )

        # past_key_values_length
        # v5 dropped `DynamicCache.from_legacy_cache` / `to_legacy_cache` /
        # `Cache.get_usable_length`; lengths come from `get_seq_length()`. A non-`Cache` argument
        # raises rather than falling back to the old `past_key_values[0][0].shape[2]`.
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError(
                "`past_key_values` must be a `transformers.cache_utils.Cache` under transformers v5 "
                f"(got {type(past_key_values).__name__}); the legacy tuple format was removed upstream."
            )
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if stage_manager.is_first_stage():
            hidden_states = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )

        # The mask is built after the embeddings because v5's helpers infer shape / dtype / device
        # from the embedding output (their second parameter is `inputs_embeds`, where v4's
        # `get_extended_attention_mask` only wanted `input_shape`). Off the first stage,
        # `hidden_states` holds the activations from the previous stage, which are equivalent for
        # that purpose.
        if self.config.is_decoder:
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
        else:
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            encoder_extended_attention_mask = None

        # inherit from bert_layer,this should be changed when we add the feature to record hidden_states
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # The old `self.encoder.gradient_checkpointing` guard would now raise: v5's `BertEncoder` is
        # a plain `nn.Module` with no such attribute. Checking pointing is handled by each
        # `BertLayer` (a `GradientCheckpointingLayer`) instead -- see the loop below.
        next_decoder_cache = () if use_cache else None

        start_idx, end_idx = stage_index[0], stage_index[1]
        # layer_outputs
        layer_outputs = hidden_states if hidden_states is not None else None

        # split the input tensor along sequence dimension
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len/TP_size, hidden_size]
        if shard_config is not None and shard_config.enable_sequence_parallelism:
            if shard_config.sequence_parallelism_mode == "split_gather":
                hidden_states = split_forward_gather_backward(
                    hidden_states,
                    dim=1,
                    process_group=shard_config.tensor_parallel_process_group,
                    fp8_communication=shard_config.fp8_communication,
                )
                if encoder_hidden_states is not None:
                    encoder_hidden_states = split_forward_gather_backward(
                        encoder_hidden_states,
                        dim=1,
                        process_group=shard_config.tensor_parallel_process_group,
                        fp8_communication=shard_config.fp8_communication,
                    )

        for idx, encoder_layer in enumerate(self.encoder.layer[start_idx:end_idx], start=start_idx):
            if stage_manager.is_first_stage() and idx == 0:
                encoder_attention_mask = encoder_extended_attention_mask

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # v5's `BertLayer.forward` takes `(hidden_states, attention_mask, encoder_hidden_states,
            # encoder_attention_mask, past_key_values, **kwargs)`: no `head_mask` and no
            # `output_attentions`, and it returns a bare tensor rather than `(hidden, [attn...])`.
            # Passing positionally would land arguments on the wrong parameters and mis-compute
            # silently, so `past_key_values` goes by name here (one whole `Cache`, indexed inside
            # `Cache.update`, not `past_key_values[idx]` per layer) and the result is used directly.
            # `hidden_states` stays positional, as `BertLayer.__call__` requires.
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
            )
            hidden_states = layer_outputs
            if use_cache:
                next_decoder_cache = past_key_values
            # `all_self_attentions` / `all_cross_attentions` stay None: this path forces
            # `output_attentions` to False above, and v5's `BertLayer` no longer returns them.

        # When sequence parallelism done, gather the output tensor in forward and split it in backward
        if shard_config is not None and shard_config.enable_sequence_parallelism:
            if shard_config.sequence_parallelism_mode == "split_gather":
                hidden_states = gather_forward_split_backward(
                    hidden_states,
                    dim=1,
                    process_group=shard_config.tensor_parallel_process_group,
                    fp8_communication=shard_config.fp8_communication,
                )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # end of a stage loop
        sequence_output = hidden_states if hidden_states is not None else None

        if stage_manager.is_last_stage():
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
            outputs = BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )
            if not return_dict:
                # The old `(sequence_output, pooled_output) + layer_outputs[1:]` would slice a
                # tensor now that `layer_outputs` is a bare tensor. `to_tuple()` follows the
                # dataclass field order instead of a hand-written literal.
                return outputs.to_tuple()
            return outputs

        # output of non-first and non-last stages: must be a dict
        else:
            # intermediate stage always return dict
            return {
                "hidden_states": hidden_states,
            }

    @staticmethod
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
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        logger = logging.get_logger(__name__)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # TODO(jianghai) left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        outputs = BertPipelineForwards.bert_model_forward(
            self.bert,
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
            hidden_states=hidden_states if hidden_states is not None else None,
            stage_index=stage_index,
            shard_config=shard_config,
        )

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
            hidden_states = outputs.get("hidden_states")

            # intermediate stage always return dict
            return {
                "hidden_states": hidden_states,
            }

    @staticmethod
    def bert_lm_head_model_forward(
        self: BertLMHeadModel,
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
        stage_manager: Optional[PipelineStageManager] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
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
        logger = logging.get_logger(__name__)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        outputs = BertPipelineForwards.bert_model_forward(
            self.bert,
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
            hidden_states=hidden_states if hidden_states is not None else None,
            stage_index=stage_index,
            shard_config=shard_config,
        )
        past_key_values = None

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
            hidden_states = outputs.get("hidden_states")
            # intermediate stage always return dict
            return {"hidden_states": hidden_states}

    @staticmethod
    def bert_for_masked_lm_forward(
        self: BertForMaskedLM,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.Tensor] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        logger = logging.get_logger(__name__)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        outputs = BertPipelineForwards.bert_model_forward(
            self.bert,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            hidden_states=hidden_states,
            stage_manager=stage_manager,
            stage_index=stage_index,
            shard_config=shard_config,
        )

        if stage_manager.is_last_stage():
            sequence_output = outputs[0]
            prediction_scores = self.cls(sequence_output)

            masked_lm_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            if not return_dict:
                output = (prediction_scores,) + outputs[2:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return MaskedLMOutput(
                loss=masked_lm_loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            hidden_states = outputs.get("hidden_states")
            return {"hidden_states": hidden_states}

    @staticmethod
    def bert_for_next_sentence_prediction_forward(
        self: BertForNextSentencePrediction,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.Tensor] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
        **kwargs,
    ):
        # -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """
        logger = logging.get_logger(__name__)

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        outputs = BertPipelineForwards.bert_model_forward(
            self.bert,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            hidden_states=hidden_states,
            stage_manager=stage_manager,
            stage_index=stage_index,
            shard_config=shard_config,
        )

        if stage_manager.is_last_stage():
            pooled_output = outputs[1]
            seq_relationship_scores = self.cls(pooled_output)

            next_sentence_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

            if not return_dict:
                output = (seq_relationship_scores,) + outputs[2:]
                return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

            return NextSentencePredictorOutput(
                loss=next_sentence_loss,
                logits=seq_relationship_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            hidden_states = outputs.get("hidden_states")
            # intermediate stage always return dict
            return {"hidden_states": hidden_states}

    @staticmethod
    def bert_for_sequence_classification_forward(
        self: BertForSequenceClassification,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.Tensor] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        logger = logging.get_logger(__name__)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        outputs = BertPipelineForwards.bert_model_forward(
            self.bert,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            hidden_states=hidden_states,
            stage_manager=stage_manager,
            stage_index=stage_index,
            shard_config=shard_config,
        )

        if stage_manager.is_last_stage():
            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

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
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            hidden_states = outputs.get("hidden_states")
            return {"hidden_states": hidden_states}

    @staticmethod
    def bert_for_token_classification_forward(
        self: BertForTokenClassification,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.Tensor] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        logger = logging.get_logger(__name__)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        outputs = BertPipelineForwards.bert_model_forward(
            self.bert,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            hidden_states=hidden_states,
            stage_manager=stage_manager,
            stage_index=stage_index,
            shard_config=shard_config,
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
        else:
            hidden_states = outputs.get("hidden_states")
            return {"hidden_states": hidden_states}

    @staticmethod
    def bert_for_multiple_choice_forward(
        self: BertForMultipleChoice,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.Tensor] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        logger = logging.get_logger(__name__)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        # in our pipeline design,input ids are copied for every stage and shouldn't be none
        # the input_ids for multiple choice model is [batch_size, num_choices, sequence_length]
        if stage_manager.is_last_stage():
            num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = BertPipelineForwards.bert_model_forward(
            self.bert,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            hidden_states=hidden_states,
            stage_manager=stage_manager,
            stage_index=stage_index,
            shard_config=shard_config,
        )
        if stage_manager.is_last_stage():
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            reshaped_logits = logits.view(-1, num_choices)

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(reshaped_logits, labels)

            if not return_dict:
                output = (reshaped_logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return MultipleChoiceModelOutput(
                loss=loss,
                logits=reshaped_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            hidden_states = outputs.get("hidden_states")
            return {"hidden_states": hidden_states}

    @staticmethod
    def bert_for_question_answering_forward(
        self: BertForQuestionAnswering,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.Tensor] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        # NOTE: the arg start_position and end_position are used only for the last stage
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        logger = logging.get_logger(__name__)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        outputs = BertPipelineForwards.bert_model_forward(
            self.bert,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            hidden_states=hidden_states,
            stage_manager=stage_manager,
            stage_index=stage_index,
            shard_config=shard_config,
        )
        if stage_manager.is_last_stage():
            sequence_output = outputs[0]

            logits = self.qa_outputs(sequence_output)
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
                output = (start_logits, end_logits) + outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output

            return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            hidden_states = outputs.get("hidden_states")
            return {"hidden_states": hidden_states}


def get_jit_fused_bert_self_output_forward():
    from transformers.models.bert.modeling_bert import BertSelfOutput

    def forward(self: BertSelfOutput, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout_add(hidden_states, input_tensor, self.dropout.p, self.dropout.training)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

    return forward


def get_jit_fused_bert_output_forward():
    from transformers.models.bert.modeling_bert import BertOutput

    def forward(self: BertOutput, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout_add(hidden_states, input_tensor, self.dropout.p, self.dropout.training)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

    return forward


# Fix the tgt_len size in sequence parallel attention. Under ``split_gather``, ``self.query`` /
# ``self.key`` / ``self.value`` are ``Linear1D_Col(seq_parallel_mode=...)``, which **gather the full
# sequence** in forward (``linear_gather_forward_reducescatter_backward``), while ``hidden_states``
# only holds this rank's shard. v5's ``BertSelfAttention.forward`` derives the output shape from
# ``hidden_states`` (``input_shape = hidden_states.shape[:-1]``), so it would reshape a
# full-sequence ``attn_output`` into a shard-shaped tensor -- the element counts do not even match.
# Taking the shape from the projection output instead is the only reason this override exists;
# everything else follows v5's own ``BertSelfAttention.forward``.
def get_bert_sequence_parallel_attention_forward(shard_config: ShardConfig):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.bert.modeling_bert import BertSelfAttention, eager_attention_forward

    def forward(
        self: BertSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # `full_shape` is derived from the projection output, not from `hidden_states` -- see above.
        projected = self.query(hidden_states)
        full_shape = (*projected.shape[:-1], -1, self.attention_head_size)
        query_layer = projected.view(*full_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*full_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*full_shape).transpose(1, 2)

        if past_key_values is not None:
            # One whole `Cache`, indexed inside `update(k, v, layer_idx)` -- no per-layer tuple
            # subscripts, and no v4 prefix-tuning branch on `past_key_value[0]`.
            current_past_key_values = past_key_values
            if isinstance(past_key_values, EncoderDecoderCache):
                current_past_key_values = past_key_values.self_attention_cache
            key_layer, value_layer = current_past_key_values.update(key_layer, value_layer, self.layer_idx)

        # `attention_mask` comes from v5's `create_bidirectional_mask` / `create_causal_mask`,
        # already in the form the attention interface consumes, so the old hand-rolled `is_causal`
        # dispatch to SDPA is gone.
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )
        # `full_shape[1]` is the gathered, full sequence length, not this rank's shard.
        attn_output = attn_output.reshape(full_shape[0], full_shape[1], -1).contiguous()
        return attn_output, attn_weights

    return forward


def bert_sequence_parallel_forward_fn(shard_config: ShardConfig):
    r"""Replacement of ``BertModel.forward`` for ``split_gather`` sequence parallelism.

    Only the sequence splitting/gathering at the two ends of the encoder is ours; the rest is kept
    line-for-line equal to v5's ``BertModel.forward`` so that the v5 contracts (mask construction,
    ``Cache``, declarative output capture) hold:

    * ``embedding_output`` is split along the sequence dim before the encoder, and
      ``sequence_output`` is gathered back before the pooler.
    * The attention mask is built from the **full** (pre-split) embeddings -- v5's mask helpers infer
      the shapes from ``inputs_embeds``, and the q/k/v projections gather the sequence back anyway,
      so attention runs over the full sequence on every rank. Building the mask after the split
      would produce a shard-sized mask that cannot broadcast against full-sequence attention.
    * ``@capture_outputs`` is re-attached: it is v5's only producer of ``attentions`` /
      ``hidden_states`` (via hooks on ``BertLayer`` / ``BertSelfAttention``). Without it a replaced
      ``BertModel.forward`` silently loses both, and ``BertForPreTraining.forward`` reads
      ``outputs.attentions`` off our return value.
    """
    from transformers.utils.generic import merge_with_config_defaults
    from transformers.utils.output_capturing import capture_outputs

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        `output_attentions` / `output_hidden_states` / `return_dict` are not read by this body under
        v5: the first two are read off the call-site kwargs by the `@capture_outputs` wrapper, which
        then injects what it recorded, and it pops the third one and calls `to_tuple()` for `False`.
        All three are kept for API shape only.
        """
        logger = logging.get_logger(__name__)

        # `head_mask` was removed from `BertModel.forward` in v5 (along with `get_head_mask`).
        # Upstream lets it ride `**kwargs` into attention; popping it here makes "passed but ignored"
        # visible instead.
        head_mask = kwargs.pop("head_mask", None)
        if head_mask is not None:
            logger.warning_once("`head_mask` was removed from `BertModel.forward` in transformers v5 and is ignored.")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError(
                "`past_key_values` must be a `transformers.cache_utils.Cache` under transformers v5 "
                f"(got {type(past_key_values).__name__}); the legacy tuple format was removed upstream."
            )
        if use_cache and past_key_values is None:
            # v5 dropped `DynamicCache.from_legacy_cache` / `to_legacy_cache` /
            # `get_usable_length`: the cache is constructed directly and lengths come from
            # `get_seq_length()`.
            past_key_values = (
                EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
                if encoder_hidden_states is not None or self.config.is_encoder_decoder
                else DynamicCache(config=self.config)
            )

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # The mask is built after the embeddings (v5's helpers need the embedding output to infer
        # shape/dtype/device) but before the split -- see the function docstring.
        if self.config.is_decoder:
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
        else:
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=attention_mask,
            )

        if encoder_attention_mask is not None:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        # split the input tensor along sequence dimension
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len/TP_size, hidden_size]
        embedding_output = split_forward_gather_backward(
            embedding_output,
            dim=1,
            process_group=shard_config.tensor_parallel_process_group,
            fp8_communication=shard_config.fp8_communication,
        )
        if encoder_hidden_states is not None:
            encoder_hidden_states = split_forward_gather_backward(
                encoder_hidden_states,
                dim=1,
                process_group=shard_config.tensor_parallel_process_group,
                fp8_communication=shard_config.fp8_communication,
            )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            **kwargs,
        )

        # v5's `BertEncoder.forward` returns only `last_hidden_state` / `past_key_values` and its
        # layers return bare tensors: `attentions` / `hidden_states` are no longer accumulated, they
        # are injected by `@capture_outputs`.
        sequence_output = encoder_outputs.last_hidden_state

        # When sequence parallelism done, gather the output tensor in forward and split it in backward
        sequence_output = gather_forward_split_backward(
            sequence_output,
            dim=1,
            process_group=shard_config.tensor_parallel_process_group,
            fp8_communication=shard_config.fp8_communication,
        )

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
        )

    return forward


def get_jit_fused_bert_intermediate_forward():
    from transformers.models.bert.modeling_bert import BertIntermediate

    from colossalai.kernel.jit.bias_gelu import GeLUFunction as JitGeLUFunction

    def forward(self: BertIntermediate, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, bias = self.dense(hidden_states)
        hidden_states = JitGeLUFunction.apply(hidden_states, bias)
        return hidden_states

    return forward
