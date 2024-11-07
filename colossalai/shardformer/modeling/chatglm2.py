""" PyTorch ChatGLM model. """

from typing import List, Optional, Tuple

import torch
import torch.utils.checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig
from colossalai.shardformer.layer import ColoAttention
from colossalai.shardformer.layer._operation import (
    all_to_all_comm,
    gather_sp_output,
    is_share_sp_tp,
    split_forward_gather_backward,
)

from ..layer import dist_cross_entropy


def get_flash_core_attention_forward():
    from .chatglm2_6b.modeling_chatglm import CoreAttention

    def forward(self: CoreAttention, query_layer, key_layer, value_layer, attention_mask):
        query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
        context_layer = ColoAttention.attention(query_layer, key_layer, value_layer, **attention_mask)
        context_layer = context_layer.permute(2, 0, 1, 3)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer

    return forward


def get_jit_fused_glm_block_forward():
    from .chatglm2_6b.modeling_chatglm import GLMBlock

    def forward(
        self: GLMBlock,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):
        # hidden_states: [s, b, h]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = self.dropout_add(attention_output, residual, self.hidden_dropout, self.training)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.dropout_add(mlp_output, residual, self.hidden_dropout, self.training)

        return output, kv_cache

    return forward


class ChatGLMPipelineForwards:
    """
    This class serves as a micro library for ChatGLM model forwards under pipeline parallelism.
    """

    @staticmethod
    def chatglm_model_forward(
        self: "ChatGLMModel",
        input_ids,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
        force_sp_output_gather: Optional[bool] = True,
    ):
        logger = logging.get_logger(__name__)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if past_key_values:
            logger.warning_once("Non-empty past_key_values is not supported for pipeline models at the moment.")
            past_key_values = None
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False
        if stage_manager.is_first_stage():
            batch_size, seq_length = input_ids.shape
            if inputs_embeds is None:
                inputs_embeds = self.embedding(input_ids)
            hidden_states = inputs_embeds
        else:
            seq_length, batch_size = hidden_states.shape[:2]
        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(
                    batch_size=batch_size,
                    device=input_ids.device,
                    dtype=inputs_embeds.dtype,
                )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask.new_ones((batch_size, self.pre_seq_len)),
                        attention_mask,
                    ],
                    dim=-1,
                )

        if shard_config.enable_flash_attention:
            mask_shape = (batch_size, 1, seq_length, seq_length)
            full_attention_mask: dict = ColoAttention.prepare_attn_kwargs(
                mask_shape,
                hidden_states.dtype,
                hidden_states.device,
                q_padding_mask=attention_mask,
                is_causal=True,
            )
        else:
            if full_attention_mask is None:
                if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                    full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # Support SP + PP
        sp_size = shard_config.sequence_parallel_size
        sp_mode = shard_config.sequence_parallelism_mode
        sp_group = shard_config.sequence_parallel_process_group
        # For generating full positions ids (the states will be gathered along the seq dim before attention fwd).
        if sp_mode != "ring_attn" and not stage_manager.is_first_stage():
            seq_length *= sp_size

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        if not past_key_values:
            past_key_values = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        if self.encoder.gradient_checkpointing and self.encoder.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        start_idx, end_idx = stage_index[0], stage_index[1]

        # Keep the input split across all PP stages
        if stage_manager.is_first_stage():
            if shard_config.enable_sequence_parallelism:
                if sp_mode == "split_gather":
                    hidden_states = split_forward_gather_backward(
                        hidden_states,
                        dim=0,
                        process_group=sp_group,
                    )
                elif shard_config.sequence_parallelism_mode == "all_to_all":
                    hidden_states = split_forward_gather_backward(
                        hidden_states,
                        dim=0,
                        process_group=shard_config.sequence_parallel_process_group,
                        grad_scale=1 / shard_config.sequence_parallel_size,
                    )

        for idx in range(start_idx, end_idx):
            layer = self.encoder._get_layer(idx)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.encoder.gradient_checkpointing and self.encoder.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    full_attention_mask,
                    rotary_pos_emb,
                    past_key_values[idx],
                    use_cache,
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    full_attention_mask,
                    rotary_pos_emb,
                    kv_cache=past_key_values[idx],
                    use_cache=use_cache,
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if stage_manager.is_last_stage():
            # final layer_norm
            if self.encoder.post_layer_norm:
                hidden_states = self.encoder.final_layernorm(hidden_states)

            # Gather seq-wise in the final output stage
            if shard_config.enable_sequence_parallelism:
                sp_mode = shard_config.sequence_parallelism_mode
                if (not shard_config.parallel_output) or force_sp_output_gather or is_share_sp_tp(sp_mode):
                    hidden_states = gather_sp_output(hidden_states, shard_config, sp_dim=0)

            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        presents,
                        all_hidden_states,
                        all_self_attentions,
                    ]
                    if v is not None
                )
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
        else:
            return {"hidden_states": hidden_states}

    @staticmethod
    def chatglm_for_conditional_generation_forward(
        self: "ChatGLMForConditionalGeneration",
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        logging.get_logger(__name__)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = ChatGLMPipelineForwards.chatglm_model_forward(
            self.transformer,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            shard_config=shard_config,
            force_sp_output_gather=False,
        )
        if stage_manager.is_last_stage():
            hidden_states = transformer_outputs[0]
            if return_last_logit:
                hidden_states = hidden_states[-1:]
            lm_logits = self.transformer.output_layer(hidden_states)
            lm_logits = lm_logits.transpose(0, 1).contiguous()

            loss = None
            if labels is not None:
                # ChatGLM doesn't have lm_head split
                enable_tp = shard_config.enable_tensor_parallelism
                shard_config.enable_tensor_parallelism = False
                loss = dist_cross_entropy(
                    labels,
                    lm_logits,
                    shard_config,
                    self.transformer.output_layer.out_features,
                    lm_logits.dtype,
                )
                shard_config.enable_tensor_parallelism = enable_tp

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
        else:
            return transformer_outputs


def get_chatglm_sequence_parallel_forward_fn(shard_config: ShardConfig, sp_mode, sp_size, sp_group):
    logger = logging.get_logger(__name__)

    def forward(
        self,
        input_ids,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        force_sp_output_gather: Optional[bool] = True,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(
                    batch_size=batch_size,
                    device=input_ids.device,
                    dtype=inputs_embeds.dtype,
                )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask.new_ones((batch_size, self.pre_seq_len)),
                        attention_mask,
                    ],
                    dim=-1,
                )
        if shard_config.enable_flash_attention:
            mask_shape = (batch_size, 1, seq_length, seq_length)
            full_attention_mask: dict = ColoAttention.prepare_attn_kwargs(
                mask_shape,
                hidden_states.dtype,
                hidden_states.device,
                q_padding_mask=attention_mask,
                is_causal=True,
            )
        else:
            if full_attention_mask is None:
                if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                    full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        if sp_mode in ["all_to_all"] and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with sp mode `{sp_mode}`. Setting `use_cache=False`..."
                )
                use_cache = False
        if sp_mode in ["all_to_all"] and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with sp mode `{sp_mode}`. Setting `use_cache=False`..."
                )
                use_cache = False
        # Run encoder.
        # [seq_len, batch_size, hidden_size] -> [seq_len/TP_size, batch_size, hidden_size]
        if sp_mode in ["split_gather"]:
            inputs_embeds = split_forward_gather_backward(
                inputs_embeds,
                dim=0,
                process_group=sp_group,
                fp8_communication=shard_config.fp8_communication,
            )
        elif sp_mode == "all_to_all":
            inputs_embeds = split_forward_gather_backward(
                inputs_embeds,
                dim=0,
                process_group=sp_group,
                grad_scale=1 / sp_size,
                fp8_communication=shard_config.fp8_communication,
            )
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        if shard_config.enable_sequence_parallelism:
            if (not shard_config.parallel_output) or force_sp_output_gather or is_share_sp_tp(sp_mode):
                hidden_states = gather_sp_output(hidden_states, shard_config, sp_dim=0)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    return forward


def get_chatglm_sequence_parallel_attention_forward(shard_config: ShardConfig, sp_mode, sp_size, sp_group):
    from .chatglm2_6b.modeling_chatglm import apply_rotary_pos_emb, split_tensor_along_last_dim

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):
        if sp_mode is not None:
            assert sp_mode in ["all_to_all", "split_gather"], "Invalid sp_mode"
            assert (sp_size is not None) and (
                sp_group is not None
            ), "Must specify sp_size and sp_group for sequence parallel"

        mixed_x_layer = self.query_key_value(hidden_states)
        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1]
                + (
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # sp: all-to-all comminucation when introducing sequence parallel
        if sp_mode == "all_to_all":
            sq, bs, _, _ = value_layer.size()

            query_layer = query_layer.reshape(sq, bs, -1)
            key_layer = key_layer.reshape(sq, bs, -1)
            value_layer = value_layer.reshape(sq, bs, -1)

            query_layer = all_to_all_comm(
                query_layer,
                sp_group,
                gather_dim=0,
                fp8_communication=shard_config.fp8_communication,
            )
            key_layer = all_to_all_comm(
                key_layer,
                sp_group,
                gather_dim=0,
                fp8_communication=shard_config.fp8_communication,
            )
            value_layer = all_to_all_comm(
                value_layer,
                sp_group,
                gather_dim=0,
                fp8_communication=shard_config.fp8_communication,
            )

            query_layer = query_layer.view(
                sq * sp_size,
                bs,
                self.num_attention_heads_per_partition // sp_size,
                self.hidden_size_per_attention_head,
            ).contiguous()

            key_layer = key_layer.view(
                sq * sp_size,
                bs,
                self.num_attention_heads_per_partition // sp_size,
                self.hidden_size_per_attention_head,
            ).contiguous()

            value_layer = value_layer.view(
                sq * sp_size,
                bs,
                self.num_attention_heads_per_partition // sp_size,
                self.hidden_size_per_attention_head,
            ).contiguous()

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(
                -1,
                -1,
                -1,
                self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
                -1,
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2]
                + (
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1,
                -1,
                -1,
                self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
                -1,
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2]
                + (
                    self.num_attention_heads_per_partition // sp_size,
                    self.hidden_size_per_attention_head,
                )
            )

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        if sp_mode == "all_to_all":
            context_layer = all_to_all_comm(
                context_layer,
                sp_group,
                gather_dim=2,
                scatter_dim=0,
                fp8_communication=shard_config.fp8_communication,
            )

        # =================
        # Output. [sq, b, h]
        # =================
        output = self.dense(context_layer)

        return output, kv_cache

    return forward


def get_flash_attention_forward_for_chat_glm_model():
    from .chatglm2_6b.modeling_chatglm import ChatGLMModel

    def forward(
        self: ChatGLMModel,
        input_ids,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(
                    batch_size=batch_size, device=input_ids.device, dtype=inputs_embeds.dtype
                )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask.new_ones((batch_size, self.pre_seq_len)), attention_mask], dim=-1
                )

        mask_shape = (batch_size, 1, seq_length, seq_length)
        full_attention_mask: dict = ColoAttention.prepare_attn_kwargs(
            mask_shape,
            inputs_embeds.dtype,
            inputs_embeds.device,
            q_padding_mask=attention_mask,
            is_causal=True,
        )

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    return forward
