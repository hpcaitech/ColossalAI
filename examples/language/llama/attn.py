import math
from types import MethodType
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

SUPPORT_XFORMERS = False
SUPPORT_FLASH2 = False
try:
    import xformers.ops as xops
    SUPPORT_XFORMERS = True
except ImportError:
    pass

try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    SUPPORT_FLASH2 = True
except ImportError:
    pass

SUPPORT_FLASH = SUPPORT_XFORMERS or SUPPORT_FLASH2

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

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

    if hasattr(self,'num_key_value_heads'):
        num_key_value_heads = self.num_key_value_heads
    else:
        num_key_value_heads = self.num_heads
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)

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
    if hasattr(self,'num_key_value_groups'):
        num_key_value_groups = self.num_key_value_groups
    else:
        num_key_value_groups = 1
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    if SUPPORT_FLASH2:
        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
    else:
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
