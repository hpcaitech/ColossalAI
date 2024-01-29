# This code is adapted from huggingface transformers: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/llama/modeling_llama.py
from typing import List, Optional, Tuple

import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
)

from colossalai.inference.flash_decoding_utils import FDIntermTensors
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
    logits = torch.mm(hidden_states, self.lm_head.weight.transpose(0, 1))
    return logits


@torch.no_grad()
def llama_model_forward(
    self: LlamaModel,
    batch: BatchInfo = None,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
):
    input_ids = batch.get_1D_inputs()
    block_tables = batch.get_block_table_tensor()

    sequence_lengths = batch.get_sequence_lengths()
    batch_size = len(sequence_lengths)
    kv_seq_len = sequence_lengths.max().item()

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
            block_tables=block_tables,
            k_cache=k_caches[layer_id],
            v_cache=v_caches[layer_id],
            is_prompts=batch.is_prompts,
            sequence_lengths=sequence_lengths,
            kv_seq_len=kv_seq_len,
            cos_sin=cos_sin,
            fd_inter_tensor=batch.fd_inter_tensor,
            output_tensor=output_tensor,
            sm_scale=sm_scale,
        )

    if batch.is_prompts:
        last_token_indexs = sequence_lengths.cumsum(dim=-1)
        hidden_states = hidden_states[last_token_indexs - 1].contiguous()
    hidden_states = self.norm(hidden_states)

    return hidden_states


@torch.no_grad()
def llama_decoder_layer_forward(
    self: LlamaDecoderLayer,
    hidden_states: torch.Tensor,
    block_tables: torch.Tensor = None,
    k_cache: torch.Tensor = None,
    v_cache: torch.Tensor = None,
    is_prompts: bool = True,
    sequence_lengths: torch.Tensor = None,
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
        block_tables=block_tables,
        k_cache=k_cache,
        v_cache=v_cache,
        is_prompts=is_prompts,
        sequence_lengths=sequence_lengths,
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
    block_tables: torch.Tensor = None,
    k_cache: torch.Tensor = None,
    v_cache: torch.Tensor = None,
    is_prompts: bool = True,
    sequence_lengths: torch.Tensor = None,
    kv_seq_len: int = 0,
    cos_sin: Tuple[torch.Tensor] = None,
    fd_inter_tensor: FDIntermTensors = None,
    output_tensor: torch.Tensor = None,
    sm_scale: int = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    query_states = torch.mm(hidden_states, self.q_proj.weight.transpose(0, 1)).view(-1, self.num_heads, self.head_dim)
    key_states = torch.mm(hidden_states, self.k_proj.weight.transpose(0, 1)).view(
        -1, self.num_key_value_heads, self.head_dim
    )
    value_states = torch.mm(hidden_states, self.v_proj.weight.transpose(0, 1)).view(
        -1, self.num_key_value_heads, self.head_dim
    )

    rotary_embedding(query_states, key_states, cos_sin[0], cos_sin[1])

    _, _, _, block_size = k_cache.shape

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

    attn_output = attn_output.view(-1, self.num_heads, self.head_dim)
    attn_output = attn_output.reshape(-1, self.hidden_size)
    attn_output = torch.mm(attn_output, self.o_proj.weight.transpose(0, 1))

    return attn_output


@torch.no_grad()
def nopad_mlp(self: LlamaMLP, hidden_states: torch.Tensor):
    gate_proj_out = torch.mm(hidden_states, self.gate_proj.weight.transpose(0, 1))
    act_out = torch.nn.functional.silu(gate_proj_out, inplace=True)
    up_proj_out = torch.mm(hidden_states, self.up_proj.weight.transpose(0, 1))
    tmp_out = act_out * up_proj_out
    return torch.mm(tmp_out, self.down_proj.weight.transpose(0, 1))
