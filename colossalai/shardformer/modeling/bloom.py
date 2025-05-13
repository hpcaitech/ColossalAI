import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    CausalLMOutputWithPast,
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
    dropout_add,
)
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer._operation import gather_forward_split_backward, split_forward_gather_backward
from colossalai.shardformer.shard import ShardConfig

from ..layer import dist_cross_entropy

logger = logging.get_logger(__name__)


def build_bloom_alibi_tensor_fn(process_group: ProcessGroup) -> torch.Tensor:
    def build_bloom_alibi_tensor(
        self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
    ) -> torch.Tensor:
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
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = torch.tensor(
            2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != num_heads:
            extra_base = torch.tensor(
                2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
                device=attention_mask.device,
                dtype=torch.float32,
            )
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_powers = torch.arange(
                1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32
            )
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
            alibi = alibi[:, offset : num_heads_per_rank + offset, :, :]
            return alibi.reshape(batch_size * num_heads_per_rank, 1, seq_length).to(dtype)
        else:
            return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)

    return build_bloom_alibi_tensor


class BloomPipelineForwards:
    """
    This class serves as a micro library for bloom pipeline forwards.
    """

    @staticmethod
    def bloom_model_forward(
        self: BloomModel,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], "BaseModelOutputWithPastAndCrossAttentions"]:
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
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # add warnings here
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False
        past_key_values = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N

        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # case: First stage of training
        if stage_manager.is_first_stage():
            # check input_ids and inputs_embeds
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            if self.gradient_checkpointing and self.training and use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)

            hidden_states = self.word_embeddings_layernorm(inputs_embeds)

            batch_size, seq_length, _ = inputs_embeds.shape
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            if cache_position is None:
                cache_position = torch.arange(past_length, past_length + seq_length, device=inputs_embeds.device)
            # initialize in the first stage and then pass to the next stage
        else:
            input_shape = hidden_states.shape[:-1]
            batch_size, seq_length = input_shape
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            if cache_position is None:
                cache_position = torch.arange(past_length, past_length + seq_length, device=hidden_states.device)

        # extra recording tensor should be generated in the first stage

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        # Compute alibi tensor: check build_alibi_tensor documentation,build for every stage
        past_length = 0
        seq_length_with_past = seq_length + past_length

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        # causal_mask is constructed every stage and its input is passed through different stages
        causal_mask = self._update_causal_mask(
            attention_mask, hidden_states, cache_position, past_key_values, output_attentions
        )

        # split the input tensor along sequence dimension
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len/TP_size, hidden_size]
        if shard_config and shard_config.enable_sequence_parallelism:
            if shard_config.sequence_parallelism_mode == "split_gather":
                hidden_states = split_forward_gather_backward(
                    hidden_states,
                    dim=1,
                    process_group=shard_config.tensor_parallel_process_group,
                    fp8_communication=shard_config.fp8_communication,
                )

        start_idx, end_idx = stage_index[0], stage_index[1]
        for i, block in enumerate(self.h[start_idx:end_idx], start=start_idx):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    causal_mask,
                    past_key_values,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                    cache_position,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=past_key_values,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                    cache_position=cache_position,
                )

            hidden_states = outputs[0]
            if use_cache:
                next_decoder_cache = outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # When sequence parallelism done, gather the output tensor in forward and split it in backward
        if shard_config and shard_config.enable_sequence_parallelism:
            if shard_config.sequence_parallelism_mode == "split_gather":
                hidden_states = gather_forward_split_backward(
                    hidden_states,
                    dim=1,
                    process_group=shard_config.tensor_parallel_process_group,
                    fp8_communication=shard_config.fp8_communication,
                )

        if stage_manager.is_last_stage():
            # Add last hidden state
            hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if stage_manager.is_last_stage():
            if not return_dict:
                return tuple(
                    v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions] if v is not None
                )

            # attention_mask is not returned ; presents = past_key_values
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
        else:
            # always return dict for imediate stage
            return {"hidden_states": hidden_states}

    @staticmethod
    def bloom_for_causal_lm_forward(
        self: BloomForCausalLM,
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
        shard_config: ShardConfig = None,
        **deprecated_arguments,
    ):
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
        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
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
            shard_config=shard_config,
        )
        past_key_values = None
        if stage_manager.is_last_stage():
            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states).contiguous()

            loss = None
            if labels is not None:
                loss = dist_cross_entropy(
                    labels,
                    lm_logits,
                    shard_config,
                    self.lm_head.out_features,
                    self.transformer.dtype,
                )

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
            hidden_states = transformer_outputs.get("hidden_states")
            return {"hidden_states": hidden_states}

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
        shard_config: ShardConfig = None,
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

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
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
            shard_config=shard_config,
        )
        past_key_values = None
        if stage_manager.is_last_stage():
            batch_size = hidden_states.shape[0]
            # update batch size
            hidden_states = transformer_outputs[0]
            logits = self.score(hidden_states)

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                    sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                    sequence_lengths = sequence_lengths % input_ids.shape[-1]
                    sequence_lengths = sequence_lengths.to(logits.device)
                else:
                    sequence_lengths = -1
                    logger.warning(
                        f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                        "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                    )

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
            hidden_states = transformer_outputs.get("hidden_states")
            return {"hidden_states": hidden_states}

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
        shard_config: ShardConfig = None,
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

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
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
            shard_config=shard_config,
        )
        past_key_values = None

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
                loss = loss_fct(
                    logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
                )

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
            hidden_states = transformer_outputs.get("hidden_states")
            return {"hidden_states": hidden_states}

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
        shard_config: ShardConfig = None,
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
        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
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


def get_jit_fused_bloom_attention_forward():
    from transformers.models.bloom.modeling_bloom import BloomAttention

    def forward(
        self: BloomAttention,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        batch_size, q_length, _ = hidden_states.shape
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        # 3 x [batch_size, num_heads, seq_length, head_dim]
        query_layer, key_layer, value_layer = self._reshape(fused_qkv)

        if layer_past is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_layer, value_layer = layer_past.update(key_layer, value_layer, self.layer_idx, cache_kwargs)

        # reshape qkv for further computations
        query_layer = query_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)
        key_layer = key_layer.reshape(batch_size * self.num_heads, -1, self.head_dim).transpose(-1, -2)
        value_layer = value_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)

        # [batch_size * num_heads, q_length, kv_length]
        attention_scores = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attn_weights = attention_scores.view(batch_size, self.num_heads, q_length, -1)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_layer.shape[-1]]
            attn_weights = attn_weights + causal_mask

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, -1)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, layer_past)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs

    return forward


def get_jit_fused_bloom_mlp_forward():
    from transformers.models.bloom.modeling_bloom import BloomMLP

    def forward(self: BloomMLP, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + F.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)
        output = self.dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)
        return output

    return forward


def get_jit_fused_bloom_gelu_forward():
    from transformers.models.bloom.modeling_bloom import BloomGelu

    from colossalai.kernel.jit.bias_gelu import GeLUFunction as JitGeLUFunction

    def forward(self: BloomGelu, x: torch.Tensor) -> torch.Tensor:
        bias = torch.zeros_like(x)
        if self.training:
            return JitGeLUFunction.apply(x, bias)
        else:
            return self.bloom_gelu_forward(x, bias)

    return forward


# Fixed the q_length args when doing the sequence parallelism in bloom model.
def get_bloom_sequence_parallel_attention_forward(shard_config: ShardConfig):
    from transformers.models.bloom.modeling_bloom import BloomAttention

    def forward(
        self: BloomAttention,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Cache] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        batch_size, q_length, _ = hidden_states.shape
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        # 3 x [batch_size, num_heads, seq_length, head_dim]
        query_layer, key_layer, value_layer = self._reshape(fused_qkv)

        if layer_past is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_layer, value_layer = layer_past.update(key_layer, value_layer, self.layer_idx, cache_kwargs)

        # reshape qkv for further computations
        query_layer = query_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)
        key_layer = key_layer.reshape(batch_size * self.num_heads, -1, self.head_dim).transpose(-1, -2)
        value_layer = value_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)

        # [batch_size * num_heads, q_length, kv_length]
        attention_scores = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )
        if shard_config.enable_sequence_parallelism:
            _, q_length, _ = query_layer.shape
        # change view to [batch_size, num_heads, q_length, kv_length]
        attn_weights = attention_scores.view(batch_size, self.num_heads, q_length, -1)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_layer.shape[-1]]
            attn_weights = attn_weights + causal_mask

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, -1)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, layer_past)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs

    return forward


def get_bloom_sequence_parallel_forward_fn(shard_config: ShardConfig):
    from transformers import BloomModel

    def forward(
        self: BloomModel,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
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
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        batch_size, seq_length, _ = inputs_embeds.shape
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        seq_length_with_past = seq_length + past_length
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + seq_length, device=inputs_embeds.device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        next_decoder_cache = None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = split_forward_gather_backward(
            hidden_states,
            dim=1,
            process_group=shard_config.tensor_parallel_process_group,
            fp8_communication=shard_config.fp8_communication,
        )

        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    causal_mask,
                    past_key_values,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                    cache_position,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=past_key_values,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                    cache_position=cache_position,
                )

            hidden_states = outputs[0]
            if use_cache:
                next_decoder_cache = outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # When sequence parallelism done, gather the output tensor in forward and split it in backward
        hidden_states = gather_forward_split_backward(
            hidden_states,
            dim=1,
            process_group=shard_config.tensor_parallel_process_group,
            fp8_communication=shard_config.fp8_communication,
        )

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions] if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    return forward


def get_lm_forward_with_dist_cross_entropy(shard_config: ShardConfig):
    from transformers import BloomForCausalLM

    def forward(
        self: BloomForCausalLM,
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        past_key_values = None
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = dist_cross_entropy(
                labels, lm_logits, shard_config, self.lm_head.out_features, self.transformer.dtype
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    return forward
