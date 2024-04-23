import math
from typing import List, Optional, Tuple

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaModel

from colossalai.inference.tensor_parallel.batch_infer_state import BatchInferState
from colossalai.kernel.triton import llama_context_attn_fwd, token_attention_fwd
from colossalai.kernel.triton.token_attention_kernel import Llama2TokenAttentionForwards

from ._utils import copy_kv_to_mem_cache

try:
    from lightllm.models.llama.triton_kernel.context_flashattention_nopad import (
        context_attention_fwd as lightllm_llama_context_attention_fwd,
    )
    from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd as llama_rotary_embedding_fwd

    HAS_LIGHTLLM_KERNEL = True
except:
    print("please install lightllm from source to run inference: https://github.com/ModelTC/lightllm")
    HAS_LIGHTLLM_KERNEL = False

try:
    from flash_attn import flash_attn_with_kvcache

    HAS_FLASH_KERNEL = True
except:
    HAS_FLASH_KERNEL = False
    print("please install flash attentiom from https://github.com/Dao-AILab/flash-attention")


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def llama_triton_context_attention(
    query_states, key_states, value_states, attn_output, infer_state, num_key_value_groups=1
):
    # if num_key_value_groups == 1:
    if HAS_LIGHTLLM_KERNEL is False:
        llama_context_attn_fwd(
            query_states,
            key_states,
            value_states,
            attn_output,
            infer_state.start_loc,
            infer_state.seq_len,
            # infer_state.cache_manager.past_key_values_length,
            infer_state.max_len_in_batch,
        )
    else:
        lightllm_llama_context_attention_fwd(
            query_states,
            key_states,
            value_states,
            attn_output,
            infer_state.start_loc,
            infer_state.seq_len,
            # infer_state.cache_manager.past_key_values_length,
            infer_state.max_len_in_batch,
        )


def llama_triton_token_attention(query_states, attn_output, infer_state, num_key_value_groups=1):
    assert HAS_LIGHTLLM_KERNEL is True, "You have to install lightllm kernel to run token attention for llama models"
    if num_key_value_groups == 1:
        token_attention_fwd(
            query_states,
            infer_state.cache_manager.key_buffer[infer_state.decode_layer_id],
            infer_state.cache_manager.value_buffer[infer_state.decode_layer_id],
            attn_output,
            infer_state.block_loc,
            infer_state.start_loc,
            infer_state.seq_len,
            # infer_state.cache_manager.past_key_values_length,
            infer_state.max_len_in_batch,
        )

    else:
        Llama2TokenAttentionForwards.token_attn(
            query_states,
            infer_state.cache_manager.key_buffer[infer_state.decode_layer_id],
            infer_state.cache_manager.value_buffer[infer_state.decode_layer_id],
            attn_output,
            infer_state.block_loc,
            infer_state.start_loc,
            infer_state.seq_len,
            # infer_state.cache_manager.past_key_values_length,
            infer_state.max_len_in_batch,
            infer_state.other_kv_index,
        )


class LlamaInferenceForwards:
    """
    This class holds forwards for llama inference.
    We intend to replace the forward methods for LlamaModel, LlamaDecoderLayer, and LlamaAttention for LlamaForCausalLM.
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
    ):
        infer_state = self.infer_state

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if infer_state.is_context_stage:
            past_key_values_length = 0
        else:
            past_key_values_length = infer_state.max_len_in_batch - 1

        # NOTE: differentiate with prefill stage
        #       block_loc require different value-assigning method for two different stage
        if use_cache and seq_length != 1:
            # NOTE assume prefill stage
            # allocate memory block
            infer_state.is_context_stage = True  # set prefill stage, notify attention layer
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
                infer_state.block_loc[:, infer_state.max_len_in_batch - 1] = infer_state.decode_mem_index
            else:
                print(f" *** Encountered allocation non-contiguous")
                print(f"    infer_state.max_len_in_batch : {infer_state.max_len_in_batch}")
                infer_state.decode_is_contiguous = False
                alloc_mem = infer_state.cache_manager.alloc(batch_size)
                infer_state.decode_mem_index = alloc_mem
                # infer_state.decode_key_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                # infer_state.decode_value_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                infer_state.block_loc[:, infer_state.max_len_in_batch - 1] = infer_state.decode_mem_index

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.repeat(batch_size, 1)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, infer_state.max_len_in_batch), dtype=torch.bool, device=inputs_embeds.device
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        infer_state.decode_layer_id = 0
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            # NOTE: modify here for passing args to decoder layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                infer_state=infer_state,
            )
            infer_state.decode_layer_id += 1
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        hidden_states = self.norm(hidden_states)
        next_cache = next_decoder_cache if use_cache else None

        # update indices
        # infer_state.block_loc[:, infer_state.max_len_in_batch-1] = infer_state.total_token_num + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        infer_state.start_loc += torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        infer_state.seq_len += 1
        infer_state.max_len_in_batch += 1

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @staticmethod
    def llama_decoder_layer_forward(
        self: LlamaDecoderLayer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        infer_state: Optional[BatchInferState] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            infer_state=infer_state,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    @staticmethod
    def llama_flash_attn_kvcache_forward(
        self: LlamaAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        infer_state: Optional[BatchInferState] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert use_cache is True, "use_cache should be set to True using this llama attention"

        bsz, q_len, _ = hidden_states.size()

        # NOTE might think about better way to handle transposed k and v
        # key_states            [bs, seq_len, num_heads, head_dim/embed_size_per_head]
        # key_states_transposed [bs, num_heads, seq_len, head_dim/embed_size_per_head]

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # NOTE might want to revise
        #   need some way to record the length of past key values cache
        #   since we won't return past_key_value_cache right now

        cos, sin = infer_state.position_cos, infer_state.position_sin

        llama_rotary_embedding_fwd(query_states.view(-1, self.num_heads, self.head_dim), cos, sin)
        llama_rotary_embedding_fwd(key_states.view(-1, self.num_key_value_heads, self.head_dim), cos, sin)

        query_states = query_states.reshape(-1, self.num_heads, self.head_dim)
        key_states = key_states.reshape(-1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(-1, self.num_key_value_heads, self.head_dim)

        if infer_state.is_context_stage:
            # first token generation
            # copy key and value calculated in current step to memory manager
            copy_kv_to_mem_cache(
                infer_state.decode_layer_id,
                key_states,
                value_states,
                infer_state.context_mem_index,
                infer_state.cache_manager,
            )
            attn_output = torch.empty_like(query_states)

            llama_triton_context_attention(
                query_states,
                key_states,
                value_states,
                attn_output,
                infer_state,
                num_key_value_groups=self.num_key_value_groups,
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
                cache_k.copy_(key_states)
                cache_v.copy_(value_states)
            else:
                # if decode is not contiguous, use triton kernel to copy key and value cache
                # k, v shape: [batch_size, num_heads, head_dim/embed_size_per_head
                copy_kv_to_mem_cache(
                    infer_state.decode_layer_id,
                    key_states,
                    value_states,
                    infer_state.decode_mem_index,
                    infer_state.cache_manager,
                )

            if HAS_LIGHTLLM_KERNEL:
                attn_output = torch.empty_like(query_states)
                llama_triton_token_attention(
                    query_states, attn_output, infer_state, num_key_value_groups=self.num_key_value_groups
                )
            else:
                self.num_heads // self.num_key_value_heads
                cache_k = infer_state.cache_manager.key_buffer[infer_state.decode_layer_id]
                cache_v = infer_state.cache_manager.value_buffer[infer_state.decode_layer_id]

                query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
                copy_cache_k = cache_k.view(bsz, -1, self.num_key_value_heads, self.head_dim)
                copy_cache_v = cache_v.view(bsz, -1, self.num_key_value_heads, self.head_dim)

                attn_output = flash_attn_with_kvcache(
                    q=query_states,
                    k_cache=copy_cache_k,
                    v_cache=copy_cache_v,
                    softmax_scale=1 / math.sqrt(self.head_dim),
                    causal=True,
                )

        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        # return past_key_value as None
        return attn_output, None, None
