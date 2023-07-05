import math
from types import MethodType
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

try:
    import xformers.ops as xops
    SUPPORT_XFORMERS = True
except ImportError:
    SUPPORT_XFORMERS = False


def llama_flash_attention(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # q, k, v is [B, H, S, K] and xformers need [B, S, H, K]. returns [B, S, H, K]
    attn_output = xops.memory_efficient_attention(query_states,
                                                  key_states,
                                                  value_states,
                                                  attn_bias=xops.LowerTriangularMask())

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def replace_xformers(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            module.forward = MethodType(llama_flash_attention, module)
