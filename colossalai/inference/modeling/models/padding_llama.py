# This code is adapted from huggingface transformers: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/llama/modeling_llama.py
from typing import List, Optional, Tuple

import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaModel

from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.inference.modeling.layers.attention import PagedAttention
from colossalai.inference.struct import BatchInfo
from colossalai.kernel.triton import (
    context_attention_unpadded,
    copy_kv_to_blocked_cache,
    flash_decoding_attention,
    get_xine_cache,
    rotary_embedding,
)
from colossalai.logging import get_dist_logger

from flash_attn.bert_padding import index_first_axis, pad_input  # noqa

logger = get_dist_logger(__name__)

try:
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    logger.warning(f"triton has not been installed yet, we will use torch to complete the attention calculation.")


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


@torch.no_grad()
def llama_causal_lm_forward(
    self: LlamaForCausalLM,
    batch: BatchInfo = None,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
):
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    hidden_states = llama_model_forward(
        self.model,
        batch=batch,
        k_caches=k_caches,
        v_caches=v_caches,
    )
    logits = self.lm_head(hidden_states)
    return logits


@torch.no_grad()
def llama_model_forward(
    self: LlamaModel,
    batch: BatchInfo = None,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
):
    input_ids = batch.get_batch_inputs()
    block_tables = batch.get_block_table_tensor()
    attention_mask = batch.get_attn_mask()

    if attention_mask is not None:
        if HAS_TRITON:
            sequence_lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
        else:
            sequence_lengths = batch.get_sequence_lengths()
    else:
        sequence_lengths = batch.get_sequence_lengths()

    batch_size, _ = input_ids.shape
    kv_seq_len = sequence_lengths.max().item()

    if attention_mask is not None:
        if batch.is_prompts:
            # Here, we generate position_ids through the input tensor, which can align with the output precision of the transformer.
            position_ids = generate_padding_position_id(attention_mask)
        else:
            position_ids = (attention_mask.sum(dim=-1) - 1).reshape(-1, 1)
    else:
        if batch.is_prompts:
            position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=batch.device)
            position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = torch.arange(kv_seq_len - 1, kv_seq_len, dtype=torch.long, device=batch.device)
            position_ids = position_ids.unsqueeze(0)

    hidden_states = self.embed_tokens(input_ids)

    cos_sin = get_xine_cache(sequence_lengths, self._cos_cached, self._sin_cached, batch.is_prompts)

    if batch.is_prompts:
        output_tensor = torch.zeros(
            (sequence_lengths.sum().item(), batch.num_heads, batch.head_dim), dtype=batch.dtype, device=batch.device
        )
    else:
        output_tensor = torch.zeros(
            (batch_size, 1, batch.num_heads, batch.head_dim), dtype=batch.dtype, device=batch.device
        )
    sm_scale = 1.0 / (batch.head_dim**0.5)

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
            kv_seq_len=kv_seq_len,
            cos_sin=cos_sin,
            fd_inter_tensor=batch.fd_inter_tensor,
            output_tensor=output_tensor,
            sm_scale=sm_scale,
        )

    hidden_states = hidden_states[:, -1, :].unsqueeze(dim=1).contiguous()
    hidden_states = self.norm(hidden_states)

    return hidden_states


@torch.no_grad()
def llama_decoder_layer_forward(
    self: LlamaDecoderLayer,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor,
    block_tables: torch.Tensor = None,
    k_cache: torch.Tensor = None,
    v_cache: torch.Tensor = None,
    is_prompts: bool = True,
    sequence_lengths: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    kv_seq_len: int = 0,
    cos_sin: Tuple[torch.Tensor] = None,
    fd_inter_tensor: FDIntermTensors = None,
    output_tensor: torch.Tensor = None,
    sm_scale: int = None,
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
        kv_seq_len=kv_seq_len,
        cos_sin=cos_sin,
        fd_inter_tensor=fd_inter_tensor,
        output_tensor=output_tensor,
        sm_scale=sm_scale,
    )

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


# Replace transformers.models.llama.modeling_llama.LlamaAttention.forward
@torch.no_grad()
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
    kv_seq_len: int = 0,
    cos_sin: Tuple[torch.Tensor] = None,
    fd_inter_tensor: FDIntermTensors = None,
    output_tensor: torch.Tensor = None,
    sm_scale: int = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    if HAS_TRITON:
        if is_prompts:
            if attention_mask is not None:
                query_states, key_states, value_states, indices = unpading_input(
                    query_states, key_states, value_states, attention_mask
                )
            else:
                query_states = query_states.view(-1, self.num_heads, self.head_dim)
                key_states = key_states.view(-1, self.num_heads, self.head_dim)
                value_states = value_states.view(-1, self.num_heads, self.head_dim)
        else:
            query_states = query_states.squeeze(dim=1)
            key_states = key_states.squeeze(dim=1)
            value_states = value_states.squeeze(dim=1)

        rotary_embedding(query_states, key_states, cos_sin[0], cos_sin[1])

        block_size = k_cache.size(-2)

        if is_prompts:
            attn_output = context_attention_unpadded(
                q=query_states,
                k=key_states,
                v=value_states,
                k_cache=k_cache,
                v_cache=v_cache,
                context_lengths=sequence_lengths,
                block_tables=block_tables,
                block_size=block_size,
                output=output_tensor,
                max_seq_len=kv_seq_len,
                sm_scale=sm_scale,
            )
            if attention_mask is not None:
                attn_output = pad_input(attn_output, indices, bsz, q_len)
        else:
            copy_kv_to_blocked_cache(key_states, k_cache, kv_lengths=sequence_lengths, block_tables=block_tables)
            copy_kv_to_blocked_cache(value_states, v_cache, kv_lengths=sequence_lengths, block_tables=block_tables)
            attn_output = flash_decoding_attention(
                q=query_states,
                k_cache=k_cache,
                v_cache=v_cache,
                kv_seq_len=sequence_lengths,
                block_tables=block_tables,
                block_size=block_size,
                max_seq_len_in_batch=kv_seq_len,
                output=output_tensor,
                mid_output=fd_inter_tensor.mid_output,
                mid_output_lse=fd_inter_tensor.mid_output_lse,
                sm_scale=sm_scale,
            )
            attn_output = attn_output.squeeze(1)
    else:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if is_prompts:
            attn_output = PagedAttention.pad_context_forward(
                query_states, key_states, value_states, k_cache, v_cache, sequence_lengths, block_tables, attention_mask
            )
        else:
            attn_output = PagedAttention.pad_decoding_forward(
                query_states, key_states, value_states, k_cache, v_cache, sequence_lengths, block_tables, attention_mask
            )

    attn_output = attn_output.view(bsz, q_len, self.num_heads, self.head_dim)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output


@torch.no_grad()
def generate_padding_position_id(attention_mask: torch.Tensor) -> torch.Tensor:
    """Generate padding position_id through attention mask.

    Args:
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

    Returns:
        torch.Tensor: The padding position_id.
    """
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


@torch.no_grad()
def unpading_input(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor):
    """Convert padding input to nopad input.

    Args:
        q (torch.Tensor): [batch_size, q_seq_len, head_num, head_dim]
        k (torch.Tensor): [batch_size, q_seq_len, head_num, head_dim]
        v (torch.Tensor): [batch_size, q_seq_len, head_num, head_dim]
        attention_mask (torch.Tensor): [batch_size, sequence_length]

    Returns:
        Tuple[torch.Tensor]: The unpad q, k, v and The index of valid data in each batch.

    """
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    batch_size, kv_seq_len, num_key_value_heads, head_dim = q.shape
    q = index_first_axis(q.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices)
    k = index_first_axis(k.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices)
    v = index_first_axis(v.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices)
    return (q, k, v, indices)
