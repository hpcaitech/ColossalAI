from typing import List, Optional, Tuple

import torch
from transformers.utils import logging

from colossalai.inference.kv_cache import BatchInferState
from colossalai.kernel.triton.token_attention_kernel import Llama2TokenAttentionForwards
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig
from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import (
    ChatGLMForConditionalGeneration,
    ChatGLMModel,
    GLMBlock,
    GLMTransformer,
    SelfAttention,
    split_tensor_along_last_dim,
)

from ._utils import copy_kv_to_mem_cache

try:
    from lightllm.models.chatglm2.triton_kernel.rotary_emb import rotary_emb_fwd as chatglm2_rotary_emb_fwd
    from lightllm.models.llama2.triton_kernel.context_flashattention_nopad import (
        context_attention_fwd as lightllm_llama2_context_attention_fwd,
    )

    HAS_LIGHTLLM_KERNEL = True
except:
    print("please install lightllm from source to run inference: https://github.com/ModelTC/lightllm")
    HAS_LIGHTLLM_KERNEL = False


def get_masks(self, input_ids, past_length, padding_mask=None):
    batch_size, seq_length = input_ids.shape
    full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
    full_attention_mask.tril_()
    if past_length:
        full_attention_mask = torch.cat(
            (
                torch.ones(batch_size, seq_length, past_length, device=input_ids.device),
                full_attention_mask,
            ),
            dim=-1,
        )

    if padding_mask is not None:
        full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
    if not past_length and padding_mask is not None:
        full_attention_mask -= padding_mask.unsqueeze(-1) - 1
    full_attention_mask = (full_attention_mask < 0.5).bool()
    full_attention_mask.unsqueeze_(1)
    return full_attention_mask


def get_position_ids(batch_size, seq_length, device):
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    return position_ids


class ChatGLM2InferenceForwards:
    """
    This class holds forwards for Chatglm2 inference.
    We intend to replace the forward methods for ChatGLMModel, ChatGLMEecoderLayer, and ChatGLMAttention.
    """

    @staticmethod
    def chatglm_for_conditional_generation_forward(
        self: ChatGLMForConditionalGeneration,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
        infer_state: Optional[BatchInferState] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        logger = logging.get_logger(__name__)

        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        # If is first stage and hidden_states is not None, go throught lm_head first
        if stage_manager.is_first_stage() and hidden_states is not None:
            if return_last_logit:
                hidden_states = hidden_states[-1:]
            lm_logits = self.transformer.output_layer(hidden_states)
            lm_logits = lm_logits.transpose(0, 1).contiguous()
            return {"logits": lm_logits}

        outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            infer_state=infer_state,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            shard_config=shard_config,
        )

        return outputs

    @staticmethod
    def chatglm_model_forward(
        self: ChatGLMModel,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        infer_state: BatchInferState = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if stage_manager.is_first_stage():
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.embedding(input_ids)
            if position_ids is None:
                position_ids = get_position_ids(batch_size, seq_length, input_ids.device)
            hidden_states = inputs_embeds
        else:
            assert hidden_states is not None, "hidden_states should not be None in non-first stage"
            seq_length, batch_size, _ = hidden_states.shape
            if position_ids is None:
                position_ids = get_position_ids(batch_size, seq_length, hidden_states.device)

        if infer_state.is_context_stage:
            past_key_values_length = 0
        else:
            past_key_values_length = infer_state.max_len_in_batch - 1

        seq_length_with_past = seq_length + past_key_values_length

        # prefill stage at first
        if seq_length != 1:
            infer_state.is_context_stage = True
            infer_state.context_mem_index = infer_state.cache_manager.alloc(infer_state.total_token_num)
            infer_state.init_block_loc(
                infer_state.block_loc, infer_state.seq_len, seq_length, infer_state.context_mem_index
            )
        else:
            infer_state.is_context_stage = False
            alloc_mem = infer_state.cache_manager.alloc_contiguous(batch_size)
            if alloc_mem is not None:
                infer_state.decode_is_contiguous = True
                infer_state.decode_mem_index = alloc_mem[0]
                infer_state.decode_mem_start = alloc_mem[1]
                infer_state.decode_mem_end = alloc_mem[2]
                infer_state.block_loc[:, seq_length_with_past - 1] = infer_state.decode_mem_index
            else:
                print(f" *** Encountered allocation non-contiguous")
                print(
                    f"    infer_state.cache_manager.past_key_values_length: {infer_state.cache_manager.past_key_values_length}"
                )
                infer_state.decode_is_contiguous = False
                alloc_mem = infer_state.cache_manager.alloc(batch_size)
                infer_state.decode_mem_index = alloc_mem
                infer_state.block_loc[:, seq_length_with_past - 1] = infer_state.decode_mem_index

        # related to rotary embedding
        if infer_state.is_context_stage:
            infer_state.position_cos = torch.index_select(self._cos_cached, 0, position_ids.view(-1)).view(
                position_ids.view(-1).shape[0], -1
            )
            infer_state.position_sin = torch.index_select(self._sin_cached, 0, position_ids.view(-1)).view(
                position_ids.view(-1).shape[0], -1
            )
        else:
            seq_len = infer_state.seq_len
            infer_state.position_cos = torch.index_select(self._cos_cached, 0, seq_len - 1).view(seq_len.shape[0], -1)
            infer_state.position_sin = torch.index_select(self._sin_cached, 0, seq_len - 1).view(seq_len.shape[0], -1)
            infer_state.other_kv_index = infer_state.block_loc[0, infer_state.max_len_in_batch - 1].item()

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
        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = get_masks(
                    self, input_ids, infer_state.cache_manager.past_key_values_length, padding_mask=attention_mask
                )

        # Run encoder.
        hidden_states = self.encoder(
            hidden_states,
            full_attention_mask,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            infer_state=infer_state,
            stage_manager=stage_manager,
            stage_index=stage_index,
            shard_config=shard_config,
        )

        # update indices
        infer_state.start_loc = infer_state.start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        infer_state.seq_len += 1
        infer_state.max_len_in_batch += 1

        return {"hidden_states": hidden_states}

    @staticmethod
    def chatglm_encoder_forward(
        self: GLMTransformer,
        hidden_states,
        attention_mask,
        kv_caches=None,
        use_cache: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
        infer_state: Optional[BatchInferState] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        infer_state.decode_layer_id = 0
        start_idx, end_idx = stage_index[0], stage_index[1]
        if kv_caches is None:
            kv_caches = tuple([None] * (end_idx - start_idx + 1))

        for idx, kv_cache in zip(range(start_idx, end_idx), kv_caches):
            layer = self.layers[idx]
            layer_ret = layer(
                hidden_states,
                attention_mask,
                kv_cache=kv_cache,
                use_cache=use_cache,
                infer_state=infer_state,
            )
            infer_state.decode_layer_id += 1

            hidden_states, _ = layer_ret

        hidden_states = hidden_states.transpose(0, 1).contiguous()

        if self.post_layer_norm and (stage_manager.is_last_stage() or stage_manager.num_stages == 1):
            # Final layer norm.
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states

    @staticmethod
    def chatglm_glmblock_forward(
        self: GLMBlock,
        hidden_states,
        attention_mask,
        kv_cache=None,
        use_cache=True,
        infer_state: Optional[BatchInferState] = None,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
            infer_state=infer_state,
        )
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output
        return output, kv_cache

    @staticmethod
    def chatglm_flash_attn_kvcache_forward(
        self: SelfAttention,
        hidden_states,
        attention_mask,
        kv_cache=None,
        use_cache=True,
        infer_state: Optional[BatchInferState] = None,
    ):
        assert use_cache is True, "use_cache should be set to True using this chatglm attention"
        # hidden_states: original :[sq, b, h] --> this [b, sq, h]
        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
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
        cos, sin = infer_state.position_cos, infer_state.position_sin

        chatglm2_rotary_emb_fwd(
            query_layer.view(-1, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head), cos, sin
        )
        if self.multi_query_attention:
            chatglm2_rotary_emb_fwd(
                key_layer.view(-1, self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head),
                cos,
                sin,
            )
        else:
            chatglm2_rotary_emb_fwd(
                key_layer.view(-1, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head),
                cos,
                sin,
            )

        # reshape q k v  to [bsz*sql, num_heads, head_dim]   2*1 ,32/2 ,128
        query_layer = query_layer.reshape(
            -1, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head
        )
        key_layer = key_layer.reshape(
            -1, self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head
        )
        value_layer = value_layer.reshape(
            -1, self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head
        )

        if infer_state.is_context_stage:
            # first token generation:
            # copy key and value calculated in current step to memory manager
            copy_kv_to_mem_cache(
                infer_state.decode_layer_id,
                key_layer,
                value_layer,
                infer_state.context_mem_index,
                infer_state.cache_manager,
            )
            attn_output = torch.empty_like(query_layer.contiguous().view(-1, self.projection_size))

            # NOTE: no bug in context attn fwd (del it )
            lightllm_llama2_context_attention_fwd(
                query_layer,
                key_layer,
                value_layer,
                attn_output.view(-1, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head),
                infer_state.start_loc,
                infer_state.seq_len,
                infer_state.max_len_in_batch,
            )

        else:
            if infer_state.decode_is_contiguous:
                # if decode is contiguous, then we copy to key cache and value cache in cache manager directly
                cache_k = infer_state.cache_manager.key_buffer[infer_state.decode_layer_id][
                    infer_state.decode_mem_start : infer_state.decode_mem_end, :, :
                ]
                cache_v = infer_state.cache_manager.value_buffer[infer_state.decode_layer_id][
                    infer_state.decode_mem_start : infer_state.decode_mem_end, :, :
                ]
                cache_k.copy_(key_layer)
                cache_v.copy_(value_layer)
            else:
                # if decode is not contiguous, use triton kernel to copy key and value cache
                # k, v shape: [batch_size, num_heads, head_dim/embed_size_per_head
                copy_kv_to_mem_cache(
                    infer_state.decode_layer_id,
                    key_layer,
                    value_layer,
                    infer_state.decode_mem_index,
                    infer_state.cache_manager,
                )

            # second token and follows
            attn_output = torch.empty_like(query_layer.contiguous().view(-1, self.projection_size))
            cache_k = infer_state.cache_manager.key_buffer[infer_state.decode_layer_id][
                : infer_state.decode_mem_end, :, :
            ]
            cache_v = infer_state.cache_manager.value_buffer[infer_state.decode_layer_id][
                : infer_state.decode_mem_end, :, :
            ]

            # ==================================
            # core attention computation is replaced by triton kernel
            # ==================================
            Llama2TokenAttentionForwards.token_attn(
                query_layer,
                cache_k,
                cache_v,
                attn_output,
                infer_state.block_loc,
                infer_state.start_loc,
                infer_state.seq_len,
                infer_state.max_len_in_batch,
                infer_state.other_kv_index,
            )

        # =================
        # Output:[b,sq, h]
        # =================
        output = self.dense(attn_output).reshape(batch_size, -1, hidden_size)

        return output, kv_cache
