#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

from colossalai.accelerator import get_accelerator
from colossalai.logging import get_dist_logger

logger = get_dist_logger()

if get_accelerator().name == "cuda":
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_kvpacked_func
    from flash_attn.ops.rms_norm import rms_norm

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
        **kwargs,
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
            output = rearrange(
                output, pattern="... h d -> ... (h d)"
            )  # (bsz, past_kv_len + q_len, num_heads * head_dim)
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

elif get_accelerator().name == "npu":
    import torch_npu

    class NPULlamaAttention(LlamaAttention):
        use_flash: bool = True

        def __init__(self, config: LlamaConfig):
            super().__init__(config)
            self.setup()

        def setup(self):
            self._softmax_scale = 1 / math.sqrt(self.head_dim)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            bsz, q_len, _ = hidden_states.size()

            if self.config.pretraining_tp > 1:
                key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
                query_slices = self.q_proj.weight.split(
                    (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
                )
                key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
                value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

                query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
                query_states = torch.cat(query_states, dim=-1)

                key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
                key_states = torch.cat(key_states, dim=-1)

                value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
                value_states = torch.cat(value_states, dim=-1)

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
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            if not self.use_flash:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

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
                    attn_weights = attn_weights + attention_mask

                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)
            else:
                attn_output, *_ = torch_npu.npu_fusion_attention(
                    query_states,
                    key_states,
                    value_states,
                    self.num_heads,
                    "BNSD",
                    atten_mask=attention_mask.bool(),
                    scale=self._softmax_scale,
                    padding_mask=None,
                    pre_tockens=65535,
                    next_tockens=0,
                    keep_prob=1.0,
                    inner_precise=0,
                )

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            if self.config.pretraining_tp > 1:
                attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                attn_output = sum(
                    [F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)]
                )
            else:
                attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

    class NPURMSNorm(LlamaRMSNorm):
        def forward(self, hidden_states):
            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]

    def replace_with_flash_attention(model: LlamaForCausalLM) -> None:
        for name, module in model.named_modules():
            if isinstance(module, LlamaAttention):
                module.__class__ = NPULlamaAttention
                module.setup()
            if isinstance(module, LlamaRMSNorm):
                module.__class__ = NPURMSNorm
