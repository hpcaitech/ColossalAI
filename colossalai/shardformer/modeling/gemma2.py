import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.utils.checkpoint
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2ForCausalLM,
    Gemma2Model,
)
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer._operation import all_to_all_comm, gather_sp_output, split_forward_gather_backward
from colossalai.shardformer.layer.utils import is_share_sp_tp, split_batch_zigzag
from colossalai.shardformer.shard import ShardConfig

from ..layer import ColoAttention, RingAttention, dist_cross_entropy

_SUPPORTED_SP_MODE = ["all_to_all", "split_gather", "ring", "ring_attn"]


class Gemma2PipelineForwards:
    """
    This class serves as a micro library for forward function substitution of Llama models
    under pipeline setting.
    """

    @staticmethod
    def gemma2_model_forward(
        self: Gemma2Model,
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
        if use_cache:
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
                batch_size, seq_length, _ = inputs_embeds.shape[:2]
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
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_length, device=device)

        seq_length_with_past = seq_length + past_seen_tokens

        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False
            
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        attn_kwargs: torch.Tensor = self._update_causal_mask(attention_mask, hidden_states, cache_position, past_key_values, output_attentions)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        start_idx, end_idx = (0, len(self.layers)) if disable_pp else (stage_index[0], stage_index[1])

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
            if idx - start_idx < num_ckpt_layers:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attn_kwargs,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attn_kwargs,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if disable_pp or stage_manager.is_last_stage():
            hidden_states = self.norm(hidden_states)
            if (not shard_config.parallel_output) or force_sp_gather or is_share_sp_tp(sp_mode):  # noqa
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
    def gemma2_for_causal_lm_forward(
        self: Gemma2ForCausalLM,
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
        outputs = Gemma2PipelineForwards.gemma2_model_forward(
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
