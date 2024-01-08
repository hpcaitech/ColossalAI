import math
from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm, apply_rotary_pos_emb, repeat_kv

SUPPORT_XFORMERS = False
SUPPORT_FLASH2 = False
try:
    import xformers.ops as xops

    SUPPORT_XFORMERS = True
except ImportError:
    pass

try:
    from flash_attn import flash_attn_func

    SUPPORT_FLASH2 = True
except ImportError:
    pass

from colossalai.utils.device import IS_NPU_AVAILABLE

SUPPORT_FLASH = SUPPORT_XFORMERS or SUPPORT_FLASH2 or IS_NPU_AVAILABLE


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
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

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

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # q, k, v is [B, H, S, K] and xformers need [B, S, H, K]. returns [B, S, H, K]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    if SUPPORT_FLASH2:
        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
    else:
        attn_output = xops.memory_efficient_attention(
            query_states, key_states, value_states, attn_bias=xops.LowerTriangularMask()
        )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    import torch_npu

    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


class NPULlamaAttention(LlamaAttention):
    fused_qkv: bool = False
    flash_attn: bool = True

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self._softmax_scale = 1 / math.sqrt(self.head_dim)
        if self.fused_qkv:
            del self.q_proj
            del self.k_proj
            del self.v_proj
            self.fused_qkv_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim + self.num_key_value_heads * self.head_dim * 2,
                bias=False,
            )
            self.split_size = (
                self.num_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        import torch_npu

        bsz, q_len, _ = hidden_states.size()

        if self.fused_qkv:
            query_states, key_states, value_states = torch.split(
                self.fused_qkv_proj(hidden_states), self.split_size, dim=-1
            )
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = npu_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if not self.flash_attn:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            # attn_weights = attn_weights + attention_mask
            if attention_mask.dtype != torch.bool:
                attention_mask.data = attention_mask.data == 0

        # upcast attention to fp32
        if not self.flash_attn:
            attn_weights = torch_npu.npu_scaled_masked_softmax(
                attn_weights, attention_mask, self._softmax_scale, False
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            # print("--------------------------------------------")
            # print("-------------------Before-------------------")
            # print("--------------------------------------------")
            # print(query_states.shape)
            # print("--------------------------------------------")
            attn_output, *_ = torch_npu.npu_fusion_attention(
                query_states,
                key_states,
                value_states,
                self.num_heads,
                "BNSD",
                atten_mask=attention_mask,
                scale=self._softmax_scale,
            )
            # print("--------------------------------------------")
            # print("-------------------After-------------------")
            # print("--------------------------------------------")

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class NPURMSNorm(LlamaRMSNorm):
    def forward(self, hidden_states):
        import torch_npu

        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]


def replace_xformers(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            if IS_NPU_AVAILABLE:
                module.__class__ = NPULlamaAttention
                module._softmax_scale = 1 / math.sqrt(module.head_dim)
            else:
                module.forward = MethodType(llama_flash_attention, module)
        # elif isinstance(module, LlamaRMSNorm):
        #     if IS_NPU_AVAILABLE:
        #         module.__class__ = NPURMSNorm
