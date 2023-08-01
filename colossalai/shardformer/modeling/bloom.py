import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.models.bloom.modeling_bloom import (
    BloomForCausalLM,
    BloomForQuestionAnswering,
    BloomForSequenceClassification,
    BloomForTokenClassification,
    BloomModel,
)
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager


def build_bloom_alibi_tensor_fn(process_group: ProcessGroup) -> torch.Tensor:

    def build_bloom_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int,
                                 dtype: torch.dtype) -> torch.Tensor:
        """
        Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
        relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
        `softmax(l+a) = softmax(l)`. Based on
        https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

        Args:
        Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
            attention_mask (`torch.Tensor`):
                Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
            num_heads (`int`, *required*):
                number of heads
            dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
                dtype of the output tensor
        """
        import math

        if dist.is_initialized():
            world_size = dist.get_world_size(process_group)
            num_heads = num_heads * world_size

        batch_size, seq_length = attention_mask.shape
        closest_power_of_2 = 2**math.floor(math.log2(num_heads))
        base = torch.tensor(2**(-(2**-(math.log2(closest_power_of_2) - 3))),
                            device=attention_mask.device,
                            dtype=torch.float32)
        powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != num_heads:
            extra_base = torch.tensor(2**(-(2**-(math.log2(2 * closest_power_of_2) - 3))),
                                      device=attention_mask.device,
                                      dtype=torch.float32)
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_powers = torch.arange(1,
                                        1 + 2 * num_remaining_heads,
                                        2,
                                        device=attention_mask.device,
                                        dtype=torch.int32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

        # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
        # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
        # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
        # => the query_length dimension will then be broadcasted correctly
        # This is more or less identical to T5's relative position bias:
        # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
        arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
        alibi = slopes[..., None] * arange_tensor
        if dist.is_initialized():
            num_heads_per_rank = int(num_heads / dist.get_world_size(process_group))
            offset = dist.get_rank(process_group) * num_heads_per_rank
            alibi = alibi.view(batch_size, num_heads, 1, seq_length)
            alibi = alibi[:, offset:num_heads_per_rank + offset, :, :]
            return alibi.reshape(batch_size * num_heads_per_rank, 1, seq_length).to(dtype)
        else:
            return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)

    return build_bloom_alibi_tensor


class BloomPipelineForwards:
    '''
    This class serves as a micro library for bloom pipeline forwards.
    '''

    @staticmethod
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
        stage_index: Optional[List[int]] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], 'BaseModelOutputWithPastAndCrossAttentions']:

        logger = logging.get_logger(__name__)

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

        start_idx, end_idx = stage_index[0], stage_index[1]
        for i, (block, layer_past) in enumerate(zip(self.h[start_idx:end_idx], past_key_values[start_idx:end_idx]),
                                                start=start_idx):
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

        if stage_manager.is_last_stage():
            if not return_dict:
                return tuple(
                    v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

            # attention_mask is not returned ; presents = past_key_values
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
        else:
            # always return dict for imediate stage
            return {'hidden_states': hidden_states}

    @staticmethod
    def bloom_for_causal_lm_forward(self: BloomForCausalLM,
                                    input_ids: Optional[torch.LongTensor] = None,
                                    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                                    attention_mask: Optional[torch.Tensor] = None,
                                    head_mask: Optional[torch.Tensor] = None,
                                    inputs_embeds: Optional[torch.Tensor] = None,
                                    labels: Optional[torch.Tensor] = None,
                                    use_cache: Optional[bool] = None,
                                    output_attentions: Optional[bool] = None,
                                    output_hidden_states: Optional[bool] = None,
                                    return_dict: Optional[bool] = None,
                                    stage_manager: Optional[PipelineStageManager] = None,
                                    hidden_states: Optional[torch.FloatTensor] = None,
                                    stage_index: Optional[List[int]] = None,
                                    **deprecated_arguments):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        logger = logging.get_logger(__name__)

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # TODO: left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
            output_attentions = False
        if output_hidden_states:
            logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
            output_hidden_states = False

        transformer_outputs = BloomPipelineForwards.bloom_model_forward(self.transformer,
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
        past_key_values = None
        all_hidden_states = None
        all_self_attentions = None
        all_cross_attentions = None
        if stage_manager.is_last_stage():
            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(lm_logits.device)
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                batch_size, seq_length, vocab_size = shift_logits.shape
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(batch_size * seq_length, vocab_size),
                                shift_labels.view(batch_size * seq_length))

            if not return_dict:
                output = (lm_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        else:
            hidden_states = transformer_outputs.get('hidden_states')
            return {'hidden_states': hidden_states}

    @staticmethod
    def bloom_for_sequence_classification_forward(
        self: BloomForSequenceClassification,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        **deprecated_arguments,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        logger = logging.get_logger(__name__)

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO: left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
            output_attentions = False
        if output_hidden_states:
            logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
            output_hidden_states = False

        transformer_outputs = BloomPipelineForwards.bloom_model_forward(
            self.transformer,
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
            stage_index=stage_index,
        )
        past_key_values = None
        all_hidden_states = None
        all_self_attentions = None
        all_cross_attentions = None
        if stage_manager.is_last_stage():
            batch_size = hidden_states.shape[0]
            #update batch size
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
                    loss = loss_fct(pooled_logits, labels)
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
    def bloom_for_token_classification_forward(
        self: BloomForTokenClassification,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        **deprecated_arguments,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        logger = logging.get_logger(__name__)

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO: left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
            output_attentions = False
        if output_hidden_states:
            logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
            output_hidden_states = False

        transformer_outputs = BloomPipelineForwards.bloom_model_forward(
            self.transformer,
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
            stage_index=stage_index,
        )
        past_key_values = None
        all_hidden_states = None
        all_self_attentions = None
        all_cross_attentions = None

        if stage_manager.is_last_stage():
            hidden_states = transformer_outputs[0]
            hidden_states = self.dropout(hidden_states)
            logits = self.classifier(hidden_states)

            loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(logits.device)
                batch_size, seq_length = labels.shape
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(batch_size * seq_length, self.num_labels),
                                labels.view(batch_size * seq_length))

            if not return_dict:
                output = (logits,) + transformer_outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        else:
            hidden_states = transformer_outputs.get('hidden_states')
            return {'hidden_states': hidden_states}

    @staticmethod
    def bloom_for_question_answering_forward(
        self: BloomForQuestionAnswering,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
    ):
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
        # TODO: left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
            output_attentions = False
        if output_hidden_states:
            logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
            output_hidden_states = False

        outputs = BloomPipelineForwards.bloom_model_forward(
            self.transformer,
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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
            hidden_states = outputs.get('hidden_states')
            return {'hidden_states': hidden_states}
