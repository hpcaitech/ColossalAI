#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_kvpacked_func
from flash_attn.ops.rms_norm import rms_norm
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

from colossalai.logging import get_dist_logger

logger = get_dist_logger()


def _prepare_decoder_attention_mask(
    self: LlamaModel,
    attention_mask: torch.BoolTensor,
    input_shape: torch.Size,
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
) -> Optional[torch.Tensor]:
    """
    Decoder attetion mask
    """
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            tensors=(
                torch.full(
                    size=(input_shape[0], past_key_values_length),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )  # (bsz, past_key_values_length + q_len)
    if attention_mask is not None and torch.all(attention_mask):
        return None  # Faster
    return attention_mask


def attention_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Re-define LLaMA-2 `LlamaAttention` forward method using flash-attention.
    """
    if output_attentions:
        logger.warning(
            "Argument `output_attentions` is not supported for flash-attention patched `LlamaAttention`, "
            "return `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        q_slicing, kv_slicing = (
            dim // self.config.pretraining_tp
            for dim in (
                self.num_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            )
        )  # `Tuple[int, int]`
        q_slices, k_slices, v_slices = (
            proj.weight.split(slicing, dim=0)
            for proj, slicing in (
                (self.q_proj, q_slicing),
                (self.k_proj, kv_slicing),
                (self.v_proj, kv_slicing),
            )
        )  # Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[torch.Tensor]]
        q, k, v = (
            torch.cat(
                [F.linear(hidden_states, slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            for slices in (q_slices, k_slices, v_slices)
        )
        # `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` of shape:
        # (bsz, q_len, num_heads * head_dim),
        # (bsz, q_len, num_key_value_heads * head_dim),
        # (bsz, q_len, num_key_value_heads * head_dim)
    else:
        q, k, v = (proj(hidden_states) for proj in (self.q_proj, self.k_proj, self.v_proj))
        # `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` of shape:
        # (bsz, q_len, num_heads * head_dim),
        # (bsz, q_len, num_key_value_heads * head_dim),
        # (bsz, q_len, num_key_value_heads * head_dim)

    # (bsz, q_len, num_heads * head_dim) -> (bsz, num_heads, q_len, head_dim);
    # (bsz, q_len, num_key_value_heads * head_dim) -> (bsz, num_key_value_heads, q_len, head_dim);
    # (bsz, q_len, num_key_value_heads * head_dim) -> (bsz, num_key_value_heads, q_len, head_dim)
    q, k, v = (
        states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        for states, num_heads in (
            (q, self.num_heads),
            (k, self.num_key_value_heads),
            (v, self.num_key_value_heads),
        )
    )
    kv_len = k.shape[-2]  # initially, `kv_len` == `q_len`
    past_kv_len = 0
    if past_key_value is not None:
        # if `past_key_value` is not None, `kv_len` > `q_len`.
        past_kv_len = past_key_value[0].shape[-2]
        kv_len += past_kv_len

    # two `torch.Tensor` objs of shape (1, 1, kv_len, head_dim)
    cos, sin = self.rotary_emb(v, seq_len=kv_len)
    # (bsz, num_heads, q_len, head_dim), (bsz, num_key_value_heads, q_len, head_dim)
    q, k = apply_rotary_pos_emb(q=q, k=k, cos=cos, sin=sin, position_ids=position_ids)
    if past_key_value is not None:
        # reuse k, v, self_attention
        k = torch.cat([past_key_value[0], k], dim=2)
        v = torch.cat([past_key_value[1], v], dim=2)

    past_key_value = (k, v) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    k = repeat_kv(hidden_states=k, n_rep=self.num_key_value_groups)
    # (bsz, num_key_value_heads, q_len, head_dim) -> (bsz, num_heads, q_len, head_dim)
    v = repeat_kv(hidden_states=v, n_rep=self.num_key_value_groups)
    # (bsz, num_key_value_heads, q_len, head_dim) -> (bsz, num_heads, q_len, head_dim)

    key_padding_mask = attention_mask
    # (bsz, num_heads, q_len, head_dim) -> (bsz, q_len, num_heads, head_dim)
    q, k, v = (states.transpose(1, 2) for states in (q, k, v))

    if past_kv_len > 0:
        q = torch.cat(
            tensors=(
                torch.full(
                    size=(bsz, past_kv_len, self.num_heads, self.head_dim),
                    fill_value=0.0,
                    dtype=q.dtype,
                    device=q.device,
                ),
                q,
            ),
            dim=1,
        )  # (bsz, past_kv_len + q_len, num_heads, head_dim)

    if key_padding_mask is None:
        # (bsz, past_kv_len + q_len, num_heads, head_dim)
        output = flash_attn_func(q=q, k=k, v=v, dropout_p=0.0, softmax_scale=None, causal=True)  # (bsz, )
        output = rearrange(output, pattern="... h d -> ... (h d)")  # (bsz, past_kv_len + q_len, num_heads * head_dim)
    else:
        q, indices, cu_q_lens, max_q_len = unpad_input(hidden_states=q, attention_mask=key_padding_mask)
        kv, _, cu_kv_lens, max_kv_len = unpad_input(
            hidden_states=torch.stack(tensors=(k, v), dim=2),
            attention_mask=key_padding_mask,
        )
        output_unpad = flash_attn_varlen_kvpacked_func(
            q=q,
            kv=kv,
            cu_seqlens_q=cu_q_lens,
            cu_seqlens_k=cu_kv_lens,
            max_seqlen_q=max_q_len,
            max_seqlen_k=max_kv_len,
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
        )
        output = pad_input(
            hidden_states=rearrange(output_unpad, pattern="nnz h d -> nnz (h d)"),
            indices=indices,
            batch=bsz,
            seqlen=past_kv_len + q_len,
        )  # (bsz, past_kv_len + q_len, num_heads * head_dim)

    if past_kv_len > 0:
        # Strip off the zero query outputs.
        output = output[:, past_kv_len:, ...]  # (bsz, q_len, num_heads * head_dim)
    output = self.o_proj(output)  # (bsz, q_len, hidden_size)
    return output, None, past_key_value


def rms_norm_forward(self: LlamaRMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Formard function for RMS Norm
    """
    return rms_norm(x=hidden_states, weight=self.weight, epsilon=self.variance_epsilon)


def replace_with_flash_attention(model: LlamaForCausalLM) -> None:
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            module.forward = MethodType(attention_forward, module)
        if isinstance(module, LlamaModel):
            module._prepare_decoder_attention_mask = MethodType(_prepare_decoder_attention_mask, module)
        if isinstance(module, LlamaRMSNorm):
            module.forward = MethodType(rms_norm_forward, module)
