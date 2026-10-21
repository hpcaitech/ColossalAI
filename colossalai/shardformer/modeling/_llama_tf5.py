import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaModel,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer._operation import all_to_all_comm, gather_sp_output, split_forward_gather_backward
from colossalai.shardformer.layer.utils import is_share_sp_tp, split_batch_zigzag
from colossalai.shardformer.shard import ShardConfig

from ..layer import ColoAttention, RingAttention, dist_cross_entropy

_SUPPORTED_SP_MODE = ["all_to_all", "split_gather", "ring", "ring_attn"]


def _get_causal_mask_input_embeddings(hidden_states, query_length: int, sequence_is_sharded: bool):
    """Return a shape-only full-sequence view when PP receives an SP shard.

    Transformers' ``create_causal_mask`` infers query length, dtype, and device
    from ``inputs_embeds``; it does not inspect embedding values. Under SP
    (all_to_all, split_gather, ring), later PP stages receive a sequence shard, while
    attention first gathers q/k/v back to the full sequence. Expanding one token as a
    view gives mask construction the full query length without copying activation data.
    """
    if not sequence_is_sharded or hidden_states.shape[1] == query_length:
        return hidden_states
    return hidden_states[:, :1, :].expand(-1, query_length, -1)


class _EagerMaskConfig:
    """Read-through view of a model config under which ``create_causal_mask`` builds an eager mask.

    Without flash attention, the policy replaces attention under SP with one that adds the mask to
    its scores before an explicit softmax, so the mask must follow eager semantics (additive float)
    whatever ``config._attn_implementation`` says: ``sdpa`` yields a boolean mask, ``flash_attention_2``
    ``None`` or a 2D padding mask, ``flex_attention`` a ``BlockMask``. ``create_causal_mask`` picks the
    mask format from ``config._attn_implementation`` only, so this view overrides that field and reads
    every other attribute from the live config, which is neither copied nor modified.
    """

    _attn_implementation = "eager"

    def __init__(self, config):
        self._config = config

    def __getattr__(self, name):
        if name == "_config":  # not yet set (e.g. during copy / unpickling): avoid infinite recursion
            raise AttributeError(name)
        return getattr(self._config, name)


class LlamaPipelineForwards:
    """
    This class serves as a micro library for forward function substitution of Llama models
    under pipeline setting.
    """

    @staticmethod
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
        cache_position: Optional[torch.LongTensor] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
        force_sp_gather: bool = True,  # Set to false only when computing cross entropy
    ):
        logger = logging.get_logger(__name__)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # This forward also replaces `LlamaModel.forward` when PP is off (plain TP / SP / FA /
        # inference), so the cache must only be disabled when PP is actually on.
        if use_cache and stage_manager is not None:
            logger.warning_once(
                "`use_cache=True` is incompatible with pipeline parallelism. Setting `use_cache=False`..."
            )
            use_cache = False

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        disable_pp = stage_manager is None
        # retrieve input_ids and inputs_embeds
        if disable_pp or stage_manager.is_first_stage():
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape[:2]
            elif inputs_embeds is not None:
                batch_size, seq_length = inputs_embeds.shape[:2]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            hidden_states = inputs_embeds
            device = hidden_states.device
        else:
            input_shape = hidden_states.shape[:-1]
            batch_size, seq_length = input_shape
            device = hidden_states.device

        # Support SP + PP
        sp_mode = shard_config.sequence_parallelism_mode
        sp_group = shard_config.sequence_parallel_process_group
        sp_size = shard_config.sequence_parallel_size
        # Generating full positions ids for modes that gather sequence before attn
        if stage_manager and (sp_mode != "ring_attn" and not stage_manager.is_first_stage()):
            seq_length *= sp_size

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                # v5 dropped `from_legacy_cache` / `to_legacy_cache`; `DynamicCache(...)`
                # accepts the same per-layer `(key, value)` tuples.
                if past_key_values is None:
                    past_key_values = DynamicCache()
                elif not isinstance(past_key_values, Cache):
                    past_key_values = DynamicCache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()
        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_length, device=device)

        seq_length_with_past = seq_length + past_seen_tokens

        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        # `output_attentions` / `output_hidden_states` stay disabled: the values accumulated
        # below are per-rank shards taken before `gather_sp_output`, i.e. not the full sequence.

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        no_split_input = disable_pp or not stage_manager.is_first_stage()
        if no_split_input and sp_mode == "ring_attn":
            _, attn_kwargs, _ = RingAttention.prepare_varlen_batch(attention_mask, sp_group)
        elif shard_config.enable_flash_attention:
            mask_shape = (batch_size, 1, seq_length_with_past, seq_length_with_past)
            attn_kwargs: dict = ColoAttention.prepare_attn_kwargs(
                mask_shape,
                hidden_states.dtype,
                hidden_states.device,
                q_padding_mask=attention_mask,
                is_causal=True,
                invert=(sp_mode != "ring_attn"),
            )
        else:
            # `allow_is_causal_skip=False` is required: the attention below uses an explicit
            # softmax, so a skipped (`None`) mask would silently attend to every position.
            causal_mask_inputs_embeds = _get_causal_mask_input_embeddings(
                hidden_states,
                seq_length,
                sequence_is_sharded=(
                    stage_manager is not None
                    and not stage_manager.is_first_stage()
                    and sp_mode in ("all_to_all", "split_gather", "ring")
                ),
            )
            # Without flash attention, the policy replaces attention only under SP, with one that adds the
            # mask to its scores; HF attention layers interpret the mask per `_attn_implementation`.
            attn_kwargs: torch.Tensor = create_causal_mask(
                config=_EagerMaskConfig(self.config) if shard_config.enable_sequence_parallelism else self.config,
                inputs_embeds=causal_mask_inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                allow_is_causal_skip=False,
            )

        # Support SP + PP. Later stages have already received the split input.
        split_input = disable_pp or stage_manager.is_first_stage()
        if split_input:
            # Ring Attention zigzag batch processing
            if sp_mode == "ring_attn":
                assert shard_config.enable_flash_attention, "Ring Attention inherently requires Flash Attention."
                if not attention_mask.bool().all():
                    hidden_states, attn_kwargs, position_ids = RingAttention.prepare_varlen_batch(
                        attention_mask, sp_group, hidden_states, position_ids
                    )
                else:
                    hidden_states, position_ids = split_batch_zigzag([hidden_states, position_ids], sp_group)

            elif is_share_sp_tp(sp_mode):
                hidden_states = split_forward_gather_backward(
                    hidden_states, 1, sp_group, fp8_communication=shard_config.fp8_communication
                )
            elif sp_mode == "all_to_all":
                hidden_states = split_forward_gather_backward(
                    hidden_states, 1, sp_group, 1 / sp_size, fp8_communication=shard_config.fp8_communication
                )

        # v5 resolves the checkpointing / cache conflict inside `GradientCheckpointingLayer`.
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        start_idx, end_idx = (0, len(self.layers)) if disable_pp else (stage_index[0], stage_index[1])
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        num_ckpt_layers = 0
        if self.gradient_checkpointing and self.training:
            num_ckpt_layers = end_idx - start_idx
            # TODO: We can replace `gradient_checkpointing_enable` fn and initialize a gradient_checkpointing (List[bool]) for each layer
            if shard_config.gradient_checkpoint_config is not None:
                num_ckpt_layers = shard_config.gradient_checkpoint_config.get_num_ckpt_layers(
                    stage=stage_manager.stage,
                    num_stages=stage_manager.num_stages,
                    num_layers=end_idx - start_idx,
                    model_chunk_id=(stage_manager.model_chunk_id if stage_manager.is_interleave else 0),
                    num_model_chunks=stage_manager.num_model_chunks,
                )
            assert num_ckpt_layers <= end_idx - start_idx
        for idx, decoder_layer in enumerate(self.layers[start_idx:end_idx], start=start_idx):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                # v5 checkpoints inside `GradientCheckpointingLayer.__call__`, based on a per-layer
                # flag; set it directly to keep ColossalAI's "first N layers only" extension.
                decoder_layer.gradient_checkpointing = idx - start_idx < num_ckpt_layers
            # Keyword names must match v5's `DecoderLayer.forward`; unknown ones land in `**kwargs`.
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attn_kwargs,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            # v5's `DecoderLayer.forward` returns a bare tensor.
            hidden_states = layer_outputs

            if use_cache:
                # v5 updates the cache in place instead of returning it.
                next_decoder_cache = past_key_values

        if disable_pp or stage_manager.is_last_stage():
            hidden_states = self.norm(hidden_states)
            if shard_config.enable_sequence_parallelism and (
                (not shard_config.parallel_output) or force_sp_gather or is_share_sp_tp(sp_mode)
            ):
                hidden_states = gather_sp_output(hidden_states, shard_config)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if disable_pp or stage_manager.is_last_stage():
            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        next_cache,
                        all_hidden_states,
                        all_self_attns,
                    ]
                    if v is not None
                )
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
        # always return dict for intermediate stage
        return {"hidden_states": hidden_states}

    @staticmethod
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
        cache_position: Optional[torch.LongTensor] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
        **kwargs,
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

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        logger = logging.get_logger(__name__)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        if shard_config.sequence_parallelism_mode == "ring_attn" and shard_config.parallel_output:
            # Split labels in a zigzag fashion too
            sp_group = shard_config.sequence_parallel_process_group
            if attention_mask.bool().all():
                labels = split_batch_zigzag(labels, sp_group, seq_dim=1, is_label=True)
            else:
                # [B, max_seqlen // sp_size]
                labels, _, _ = RingAttention.prepare_varlen_batch(attention_mask, sp_group, labels, is_label=True)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = LlamaPipelineForwards.llama_model_forward(
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
            cache_position=cache_position,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            shard_config=shard_config,
            force_sp_gather=False,
        )
        past_key_values = None

        disable_pp = stage_manager is None
        if disable_pp or stage_manager.is_last_stage():
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            loss = None
            if labels is not None:
                loss = dist_cross_entropy(labels, logits, shard_config, self.lm_head.out_features, self.model.dtype)

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
            hidden_states = outputs.get("hidden_states")
            return {"hidden_states": hidden_states}

    @staticmethod
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
        cache_position: Optional[torch.LongTensor] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
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
        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        transformer_outputs = LlamaPipelineForwards.llama_model_forward(
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
            cache_position=cache_position,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            shard_config=shard_config,
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
            hidden_states = transformer_outputs.get("hidden_states")
            return {"hidden_states": hidden_states}


def get_llama_flash_attention_forward(shard_config: ShardConfig, sp_mode=None, sp_size=None, sp_group=None):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[Union[torch.Tensor, Dict]] = None,
        # v5 renamed this argument to the plural `past_key_values`; the old name would be
        # swallowed by `**kwargs` and the cache silently dropped.
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if sp_mode is not None:
            assert sp_mode in _SUPPORTED_SP_MODE, f"SP mode {sp_mode} is not supported by {type(self)} yet"
            assert (sp_size is not None) and (
                sp_group is not None
            ), "Must specify sp_size and sp_group for sequence parallel"
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        # v5's attention has no `num_heads` attribute; it is only written by the TP and
        # SP-all_to_all policies, so read it defensively.
        num_heads = getattr(self, "num_heads", None) or self.config.num_attention_heads
        # sp: modify sp_len when sequence parallel mode is ring
        if is_share_sp_tp(sp_mode):
            q_len *= sp_size

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # sp: all-to-all comminucation when introducing sequence parallel
        if sp_mode == "all_to_all":
            query_states = all_to_all_comm(query_states, sp_group, fp8_communication=shard_config.fp8_communication)
            key_states = all_to_all_comm(key_states, sp_group, fp8_communication=shard_config.fp8_communication)
            value_states = all_to_all_comm(value_states, sp_group, fp8_communication=shard_config.fp8_communication)
            bsz, q_len, _ = query_states.size()

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_values is not None and self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # v5 removed `Cache.get_usable_length`; `update` returns k/v including history.
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            kv_seq_len = key_states.shape[-2]

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if sp_mode == "ring_attn":
            attn_output = RingAttention.attention(
                query_states,
                key_states,
                value_states,
                sp_axis=shard_config.sp_axis,
                **attention_mask,
                inner_ring_size=shard_config.inner_ring_size,
                pg_mesh=shard_config.pg_mesh,
            )

        elif shard_config.enable_flash_attention:
            assert isinstance(attention_mask, dict), "Flash Attention Error: attention_mask should be a dict."
            attn_output = ColoAttention.attention(query_states, key_states, value_states, **attention_mask)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

        attn_output = attn_output.transpose(1, 2).contiguous()
        # sp: all-to-all comminucation when introducing sequence parallel
        if sp_mode == "all_to_all":
            attn_output = attn_output.reshape(bsz, q_len, num_heads * self.head_dim)
            attn_output = all_to_all_comm(
                attn_output, sp_group, scatter_dim=1, gather_dim=2, fp8_communication=shard_config.fp8_communication
            )
        else:
            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights

    return forward
