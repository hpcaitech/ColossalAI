# This code is adapted from huggingface transformers: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/llama/modeling_llama.py
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    repeat_kv,
)

from colossalai.inference.modeling.layers.attention import convert_kvcache, copy_to_cache
from colossalai.inference.struct import BatchInfo

from flash_attn.bert_padding import index_first_axis  # noqa


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


def llama_causal_lm_forward(
    self: LlamaForCausalLM,
    batch: BatchInfo = None,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
    padding_id: int = None,
):
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    hidden_states = llama_model_forward(
        self.model,
        batch=batch,
        k_caches=k_caches,
        v_caches=v_caches,
        padding_id=padding_id,
    )
    logits = self.lm_head(hidden_states)
    return logits


def llama_model_forward(
    self: LlamaModel,
    batch: BatchInfo = None,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
    padding_id: int = None,
):
    input_ids = batch.get_batch_inputs()
    block_tables = batch.get_block_table_tensor()
    sequence_lengths = batch.get_sequence_lengths()

    attention_mask = batch.get_attn_mask(padding_id)

    if batch.is_prompts:
        # Here, we generate position_ids through the input tensor, which can align with the output precision of the transformer.
        position_ids = generate_padding_position_id(attention_mask)
    else:
        position_ids = (attention_mask.sum(dim=-1) - 1).reshape(-1, 1)

    hidden_states = self.embed_tokens(input_ids)

    for layer_id, decoder_layer in enumerate(self.layers):
        hidden_states = decoder_layer(
            hidden_states,
            position_ids=position_ids,
            block_tables=block_tables,
            k_cache=k_caches[layer_id],
            v_cache=v_caches[layer_id],
            is_prompts=batch.is_prompts,
            sequence_lengths=sequence_lengths,
            attention_mask=attention_mask,
        )

    hidden_states = self.norm(hidden_states)

    return hidden_states


def llama_decoder_layer_forward(
    self: LlamaDecoderLayer,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor,
    block_tables: torch.Tensor = None,
    k_cache: torch.Tensor = None,
    v_cache: torch.Tensor = None,
    is_prompts: bool = True,
    sequence_lengths: int = None,
    attention_mask: torch.Tensor = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states = self.self_attn(
        hidden_states=hidden_states,
        position_ids=position_ids,
        block_tables=block_tables,
        k_cache=k_cache,
        v_cache=v_cache,
        is_prompts=is_prompts,
        sequence_lengths=sequence_lengths,
        attention_mask=attention_mask,
    )

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


# Replace transformers.models.llama.modeling_llama.LlamaAttention.forward
def llama_attn_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor,
    block_tables: torch.Tensor = None,
    k_cache: torch.Tensor = None,
    v_cache: torch.Tensor = None,
    is_prompts: bool = True,
    sequence_lengths: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = sequence_lengths[0].item()

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if is_prompts:
        attn_output = pad_context_forward(
            query_states, key_states, value_states, k_cache, v_cache, sequence_lengths, block_tables, attention_mask
        )
    else:
        attn_output = pad_decoding_forward(
            query_states, key_states, value_states, k_cache, v_cache, sequence_lengths, block_tables, attention_mask
        )

    attn_output = attn_output.view(bsz, q_len, self.num_heads, self.head_dim)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output


def generate_padding_position_id(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def unpading_input(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    batch_size, kv_seq_len, num_key_value_heads, head_dim = q.shape
    q = index_first_axis(q.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices)
    k = index_first_axis(k.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices)
    v = index_first_axis(v.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices)
    return (q, k, v, indices, seqlens)


def pad_decoding_forward(
    query: torch.Tensor,  # [bsz, 1, num_heads, head_size]
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,  # [num_blocks, num_heads, head_size, block_size]
    v_cache: torch.Tensor,
    lengths: torch.Tensor,  # [num_seqs]: input_lengths + output_lengths
    block_tables: torch.Tensor,  # [num_seqs,max_blocks_per_sequence]
    attn_mask: torch.Tensor = None,
):
    bsz, query_length, num_heads, head_size = query.shape
    seq_len = max(lengths)

    copy_to_cache(key, k_cache, lengths=lengths, block_tables=block_tables, type="decoding")
    copy_to_cache(value, v_cache, lengths=lengths, block_tables=block_tables, type="decoding")

    key = convert_kvcache(k_cache, lengths, block_tables)  # bsz, seqlen,
    value = convert_kvcache(v_cache, lengths, block_tables)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_size)
    if attn_weights.size() != (bsz, num_heads, 1, seq_len):
        raise ValueError(f"Got wrong attn_weights, should be in shape {(bsz,num_heads,1,seq_len)}.")

    if attn_mask is not None:
        padding_mask = AttentionMaskConverter._expand_mask(attn_mask, query.dtype, query_length)

    attn_mask = AttentionMaskConverter._make_causal_mask(
        (bsz, query_length), query.dtype, query.device, past_key_values_length=seq_len - query_length
    )

    if padding_mask is not None:
        attn_mask = attn_mask.masked_fill(padding_mask.bool(), torch.finfo(query.dtype).min)

    attn_weights += attn_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)

    if attn_output.size() != (bsz, num_heads, 1, head_size):
        raise ValueError(f"Got wrong attn_output, should be in shape {(bsz,num_heads,1,head_size)}.")
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, 1, -1)

    return attn_output


def pad_context_forward(
    q: torch.Tensor,  # [batch_size, seq_len, num_heads, head_size]
    k: torch.Tensor,  # [batch_size, seq_len, num_kv_heads, head_size]
    v: torch.Tensor,
    k_cache: torch.Tensor,  # [num_blocks, num_heads, head_size, block_size]
    v_cache: torch.Tensor,
    context_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs,max_blocks_per_sequence]
    attn_mask: torch.Tensor = None,
):
    # Firt, do shape verification
    bsz, seq_len, num_heads, head_size = q.shape
    num_kv_heads = k.shape[-2]
    assert num_heads % num_kv_heads == 0, "num_kv_heads should be divisible by num_heads"
    num_kv_groups = num_heads // num_kv_heads
    block_size = k_cache.shape[-1]
    assert q.shape[0] == k.shape[0] == v.shape[0] == block_tables.shape[0]
    block_tables.shape[-1] * block_size

    # Copy kv to memory(rotary embedded)
    copy_to_cache(k, k_cache, lengths=context_lengths, block_tables=block_tables)
    copy_to_cache(v, v_cache, lengths=context_lengths, block_tables=block_tables)

    q = q.transpose(1, 2)
    k = repeat_kv(k.transpose(1, 2), num_kv_groups)
    v = repeat_kv(v.transpose(1, 2), num_kv_groups)

    attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_size)

    if attn_mask is not None:
        padding_mask = AttentionMaskConverter._expand_mask(attn_mask, q.dtype, seq_len)

    attn_mask = AttentionMaskConverter._make_causal_mask(
        (bsz, seq_len), q.dtype, q.device, past_key_values_length=seq_len - seq_len
    )

    if padding_mask is not None:
        attn_mask = attn_mask.masked_fill(padding_mask.bool(), torch.finfo(q.dtype).min)

    if attn_weights.size() != (bsz, num_heads, seq_len, seq_len):
        raise ValueError(f"Got wrong attn_weights, should be in shape {(bsz,num_heads,seq_len,seq_len)}.")
    if attn_mask is not None:
        attn_weights += attn_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)

    if attn_output.size() != (bsz, num_heads, seq_len, head_size):
        raise ValueError(f"Got wrong attn_output, should be in shape {(bsz,num_heads,seq_len,head_size)}.")

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)

    del attn_weights

    return attn_output
