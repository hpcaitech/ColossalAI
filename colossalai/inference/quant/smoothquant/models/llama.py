# Code modified from smoothquant: https://github.com/mit-han-lab/smoothquant

import math
import os
import types
from collections import defaultdict
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch import nn
from torch_int.nn.bmm import BMM_S8T_S8N_F32T, BMM_S8T_S8N_S8T
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LLAMA_INPUTS_DOCSTRING,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    repeat_kv,
    rotate_half,
    LlamaRotaryEmbedding,
)
from transformers.utils import add_start_docstrings_to_model_forward

from colossalai.kernel.triton import (
    int8_rotary_embedding_fwd,
    smooth_llama_context_attn_fwd,
    smooth_token_attention_fwd,
)
import torch.nn.functional as F

from .linear import W8A8B8O8Linear, W8A8BFP32O32LinearSiLU, W8A8BFP32OFP32Linear


class LLamaSmoothquantAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )

        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        self.k_proj = W8A8B8O8Linear(hidden_size, hidden_size)
        self.v_proj = W8A8B8O8Linear(hidden_size, hidden_size)
        self.q_proj = W8A8B8O8Linear(hidden_size, hidden_size)
        self.o_proj = W8A8BFP32OFP32Linear(hidden_size, hidden_size)

        self.register_buffer("q_output_scale", torch.tensor([1.0]))
        self.register_buffer("k_output_scale", torch.tensor([1.0]))
        self.register_buffer("v_output_scale", torch.tensor([1.0]))
        self.register_buffer("q_rotary_output_scale", torch.tensor([1.0]))
        self.register_buffer("k_rotary_output_scale", torch.tensor([1.0]))
        self.register_buffer("out_input_scale", torch.tensor([1.0]))
        self.register_buffer("attn_input_scale", torch.tensor([1.0]))

        self._init_rope()
        self.num_key_value_heads = num_heads
    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=2048,
            base=10000.0,
        )
            
    @staticmethod
    def pack(
        module: LlamaAttention,
        attn_input_scale: float,
        q_output_scale: float,
        k_output_scale: float,
        v_output_scale: float,
        q_rotary_output_scale: float,
        k_rotary_output_scale: float,
        out_input_scale: float,
    ):
        int8_module = LLamaSmoothquantAttention(module.hidden_size, module.num_heads)
        # self.register_buffer("attn_input_scale", torch.tensor([1.0]))
        int8_module.attn_input_scale = torch.tensor(attn_input_scale)

        int8_module.q_output_scale = torch.tensor(q_output_scale)
        int8_module.k_output_scale = torch.tensor(k_output_scale)
        int8_module.v_output_scale = torch.tensor(v_output_scale)

        int8_module.q_rotary_output_scale = torch.tensor(q_rotary_output_scale)
        int8_module.k_rotary_output_scale = torch.tensor(k_rotary_output_scale)

        int8_module.q_proj = W8A8B8O8Linear.from_float(module.q_proj, attn_input_scale, q_output_scale)
        int8_module.k_proj = W8A8B8O8Linear.from_float(module.k_proj, attn_input_scale, k_output_scale)
        int8_module.v_proj = W8A8B8O8Linear.from_float(module.v_proj, attn_input_scale, v_output_scale)
        int8_module.o_proj = W8A8BFP32OFP32Linear.from_float(module.o_proj, out_input_scale)


        # int8_module.q_proj = module.q_proj
        # int8_module.k_proj = module.k_proj
        # int8_module.v_proj = module.v_proj
        # int8_module.o_proj = module.o_proj
        int8_module.out_input_scale = torch.tensor(out_input_scale)

        return int8_module

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        cos = rotary_emb[0]
        sin = rotary_emb[1]

        int8_rotary_embedding_fwd(
            query_states.view(-1, self.num_heads, self.head_dim),
            cos,
            sin,
            self.q_output_scale.item(),
            self.q_rotary_output_scale.item(),
        )
        int8_rotary_embedding_fwd(
            key_states.view(-1, self.num_heads, self.head_dim),
            cos,
            sin,
            self.k_output_scale.item(),
            self.k_rotary_output_scale.item(),
        )

        if past_key_value is None:
            proj_shape = (bsz*q_len, self.num_heads, self.head_dim)

            query_states = query_states.view(*proj_shape)
            key_states = key_states.view(*proj_shape)
            value_states = value_states.view(*proj_shape)

            # attn_output = torch.empty(bsz*q_len, self.num_heads, self.head_dim, dtype=torch.float16, device="cuda")
            attn_output = torch.empty_like(query_states)

            b_start_loc = torch.zeros((bsz,), dtype=torch.int32, device="cuda")
            b_seq_len = torch.ones((bsz,), dtype=torch.int32, device="cuda")

            b_seq_len[0] = q_len

            for i in range(1, bsz):
                b_start_loc[i] = b_start_loc[i - 1] + b_seq_len[i - 1]

            smooth_llama_context_attn_fwd(
                query_states,
                key_states,
                value_states,
                attn_output,
                self.q_rotary_output_scale.item(),
                self.k_rotary_output_scale.item(),
                self.v_output_scale.item(),
                self.out_input_scale.item(),
                b_start_loc,
                b_seq_len,
                q_len,
            )
            if use_cache:
                past_key_value = (
                    key_states.view(bsz, q_len, -1, self.head_dim),
                    value_states.view(bsz, q_len, -1, self.head_dim),
                )
        else:
            total_seq_len = past_key_value[0].shape[1] + q_len
            key_states = torch.cat([past_key_value[0], key_states.view(bsz, q_len, -1, self.head_dim)], dim=1)
            value_states = torch.cat([past_key_value[1], value_states.view(bsz, q_len, -1, self.head_dim)], dim=1)

            proj_shape = (bsz * q_len, -1, self.head_dim)
            kv_shape = (bsz * total_seq_len, -1, self.head_dim)
            query_states = query_states.view(*proj_shape)
            key_states = key_states.view(*kv_shape)
            value_states = value_states.view(*kv_shape)
            attn_output = torch.empty_like(query_states)

            b_start_loc = torch.arange(
                start=0, end=bsz * total_seq_len, step=total_seq_len, dtype=torch.int, device="cuda"
            )
            b_seq_len = torch.full([bsz], q_len, dtype=torch.int, device="cuda") * total_seq_len
            block_loc = torch.arange(total_seq_len, dtype=torch.int, device="cuda").expand(bsz, -1)
            smooth_token_attention_fwd(
                query_states,
                key_states,
                value_states,
                attn_output,
                self.q_rotary_output_scale.item(),
                self.k_rotary_output_scale.item(),
                self.v_output_scale.item(),
                self.attn_output_scale.item(),
                block_loc,
                b_start_loc,
                b_seq_len,
                total_seq_len,
            )
            if use_cache:
                past_key_value = (
                    key_states.view(bsz, total_seq_len, -1, self.head_dim),
                    value_states.view(bsz, total_seq_len, -1, self.head_dim),
                )

        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaLayerNormQ(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.input_scale = 1.0
        self.variance_epsilon = eps
        self.register_buffer('weight', torch.ones(dim, dtype=torch.float32))

    def forward(self, x):

        input_dtype = x.dtype
        hidden_states = x.to(torch.float32)
        ln_output_fp = torch.nn.functional.layer_norm(
            x, x.shape[-1:], self.weight, None, self.variance_epsilon)
        ln_output_int8 = ln_output_fp.round().clamp(-128, 127).to(torch.int8)
        return ln_output_int8


    @staticmethod
    def from_float(module: torch.nn.LayerNorm, output_scale: float):
        assert module.weight.shape[0] == module.weight.numel()
        # assert module.bias.shape[0] == module.bias.numel()
        q_module = LlamaLayerNormQ(module.weight.shape[0], module.variance_epsilon )
        q_module.weight = module.weight / output_scale
        # q_module.bias = module.bias / output_scale
        return q_module

class LlamaSmoothquantMLP(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.gate_proj = W8A8BFP32O32LinearSiLU(hidden_size, intermediate_size)
        self.up_proj = W8A8BFP32OFP32Linear(hidden_size, intermediate_size)
        self.down_proj = W8A8BFP32OFP32Linear(intermediate_size, hidden_size)
        self.register_buffer("down_proj_input_scale", torch.tensor([1.0]))

    @staticmethod
    def pack(
        mlp_module: LlamaMLP,
        gate_proj_input_scale: float,
        up_proj_input_scale: float,
        down_proj_input_scale: float,
    ):
        int8_module = LlamaSmoothquantMLP(
            mlp_module.intermediate_size,
            mlp_module.hidden_size,
        )

        int8_module.gate_proj = W8A8BFP32O32LinearSiLU.from_float(mlp_module.gate_proj, gate_proj_input_scale)
        int8_module.up_proj = W8A8BFP32OFP32Linear.from_float(mlp_module.up_proj, up_proj_input_scale)
        int8_module.down_proj = W8A8BFP32OFP32Linear.from_float(mlp_module.down_proj, down_proj_input_scale)
        int8_module.down_proj_input_scale = torch.tensor(down_proj_input_scale)
        return int8_module

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        x_shape = hidden_states.shape
        gate_out = self.gate_proj(hidden_states)
        up_out = self.up_proj(hidden_states)
        inter_out = gate_out * up_out
        inter_out = inter_out.div_(self.down_proj_input_scale.item()).round().clamp(-128, 127).to(torch.int8)
        down_out = self.down_proj(inter_out)
        down_out = down_out.view(*x_shape[:-1], -1)
        return down_out


class LlamaSmoothquantDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LLamaSmoothquantAttention(config.hidden_size, config.num_attention_heads)

        self.mlp = LlamaSmoothquantMLP(config.intermediate_size, config.hidden_size)
        self.input_layernorm = LlamaLayerNormQ(config.hidden_size, eps=config.rms_norm_eps)

        self.post_attention_layernorm = LlamaLayerNormQ(config.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def pack(
        module: LlamaDecoderLayer,
        attn_input_scale: float,
        q_output_scale: float,
        k_output_scale: float,
        v_output_scale: float,
        q_rotary_output_scale: float,
        k_rotary_output_scale: float,
        out_input_scale: float,
        gate_input_scale: float,
        up_input_scale: float,
        down_input_scale: float,
    ):
        config = module.self_attn.config
        int8_decoder_layer = LlamaSmoothquantDecoderLayer(config)

        int8_decoder_layer.input_layernorm = LlamaLayerNormQ.from_float(module.input_layernorm,  attn_input_scale)
        int8_decoder_layer.self_attn = LLamaSmoothquantAttention.pack(
            module.self_attn,
            attn_input_scale,
            q_output_scale,
            k_output_scale,
            v_output_scale,
            q_rotary_output_scale,
            k_rotary_output_scale,
            out_input_scale,
        )


        # int8_decoder_layer.input_layernorm = module.input_layernorm
        # int8_decoder_layer.self_attn = module.self_attn


        int8_decoder_layer.post_attention_layernorm = LlamaLayerNormQ.from_float(module.post_attention_layernorm,  gate_input_scale)

        int8_decoder_layer.mlp = LlamaSmoothquantMLP.pack(
            module.mlp,
            gate_input_scale,
            up_input_scale,
            down_input_scale,
        )

        # int8_decoder_layer.post_attention_layernorm = module.post_attention_layernorm
        # int8_decoder_layer.mlp = module.mlp


        return int8_decoder_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
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


class LlamaApplyRotary(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        x_embed = (x * cos) + (rotate_half(x) * sin)

        return x_embed


def llama_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
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
    query_states = self.q_apply_rotary(query_states, cos, sin, position_ids)
    key_states = self.k_apply_rotary(key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

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
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def init_to_get_rotary(config, base=10000, use_elem=False):
    """
    This function initializes the rotary positional embedding, it is compatible for all models and is called in ShardFormer
    Args:
        base : calculation arg
        use_elem : activated when using chatglm-based models
    """
    config.head_dim_ = config.hidden_size // config.num_attention_heads
    if not hasattr(config, "rope_scaling"):
        rope_scaling_factor = 1.0
    else:
        rope_scaling_factor = config.rope_scaling.factor if config.rope_scaling is not None else 1.0

    if hasattr(config, "max_sequence_length"):
        max_seq_len = config.max_sequence_length
    elif hasattr(config, "max_position_embeddings"):
        max_seq_len = config.max_position_embeddings * rope_scaling_factor
    else:
        max_seq_len = 2048 * rope_scaling_factor
    base = float(base)

    # NTK  ref: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    try:
        ntk_alpha = float(os.environ.get("INFER_NTK_ALPHA", 1))
        assert ntk_alpha >= 1
        if ntk_alpha > 1:
            print(f"Note: NTK enabled, alpha set to {ntk_alpha}")
        max_seq_len *= ntk_alpha
        base = base * (ntk_alpha ** (config.head_dim_ / (config.head_dim_ - 2)))  # Base change formula
    except:
        pass

    n_elem = config.head_dim_
    if use_elem:
        n_elem //= 2

    inv_freq = 1.0 / (base ** (torch.arange(0, n_elem, 2, device="cpu", dtype=torch.float32) / n_elem))
    t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
    freqs = torch.outer(t, inv_freq)

    _cos_cached = torch.cos(freqs).to(torch.float)
    _sin_cached = torch.sin(freqs).to(torch.float)
    return _cos_cached, _sin_cached


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if past_key_values_length == 0:
        position_cos = torch.index_select(self._cos_cached, 0, position_ids.view(-1)).view(
            position_ids.view(-1).shape[0], -1
        )
        position_sin = torch.index_select(self._sin_cached, 0, position_ids.view(-1)).view(
            position_ids.view(-1).shape[0], -1
        )
    else:
        position_cos = torch.index_select(self._cos_cached, 0, position_ids.view(-1)).view(batch_size, -1)
        position_sin = torch.index_select(self._sin_cached, 0, position_ids.view(-1)).view(batch_size, -1)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                rotary_emb=(position_cos, position_sin),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def convert_llama_to_smoothquant(
    llama_model,
    tokenizer,
    dataset_path,
    num_samples=512,
    seq_len=512,
):
    llama_config = llama_model.config

    llama_model.eval()
    device = next(llama_model.parameters()).device
    # print("model:", llama_model)
    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(act_dict[name]["input"], x.detach().abs().max().item())
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(act_dict[name]["output"], y.detach().abs().max().item())

    for name, m in llama_model.named_modules():
        if isinstance(m, LlamaAttention):
            setattr(m, "q_apply_rotary", LlamaApplyRotary())
            setattr(m, "k_apply_rotary", LlamaApplyRotary())
            m.forward = types.MethodType(llama_decoder_layer_forward, m)

    hooks = []
    for name, m in llama_model.named_modules():
        if isinstance(m, LlamaApplyRotary):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)
    for i in pbar:
        input_ids = tokenizer(
            dataset["rows"][0][i]["row"]["text"],
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
        ).input_ids.to(device)
        llama_model(input_ids)
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()

    decoder_layer_scales = []

    for idx in range(llama_config.num_hidden_layers):
        scale_dict = {}
        scale_dict["attn_input_scale"] = act_dict[f"model.layers.{idx}.self_attn.q_proj"]["input"] / 127
        scale_dict["q_output_scale"] = act_dict[f"model.layers.{idx}.self_attn.q_proj"]["output"] / 127
        scale_dict["k_output_scale"] = act_dict[f"model.layers.{idx}.self_attn.k_proj"]["output"] / 127
        scale_dict["v_output_scale"] = act_dict[f"model.layers.{idx}.self_attn.v_proj"]["output"] / 127

        scale_dict["q_rotary_output_scale"] = act_dict[f"model.layers.{idx}.self_attn.q_apply_rotary"]["output"] / 127

        scale_dict["k_rotary_output_scale"] = act_dict[f"model.layers.{idx}.self_attn.k_apply_rotary"]["output"] / 127

        scale_dict["out_input_scale"] = act_dict[f"model.layers.{idx}.self_attn.o_proj"]["input"] / 127
        # mlp scales
        scale_dict["gate_input_scale"] = act_dict[f"model.layers.{idx}.mlp.gate_proj"]["input"] / 127
        scale_dict["up_input_scale"] = act_dict[f"model.layers.{idx}.mlp.up_proj"]["input"] / 127
        scale_dict["down_input_scale"] = act_dict[f"model.layers.{idx}.mlp.down_proj"]["input"] / 127

        decoder_layer_scales.append(scale_dict)

    for i, layer in enumerate(llama_model.model.layers):
        orig_layer = layer
        llama_model.model.layers[i] = LlamaSmoothquantDecoderLayer.pack(orig_layer, **decoder_layer_scales[i])

    llama_model.model.forward = types.MethodType(llama_model_forward, llama_model.model)

    cos, sin = init_to_get_rotary(llama_config)
    llama_model.model.register_buffer("_cos_cached", cos)
    llama_model.model.register_buffer("_sin_cached", sin)
    return decoder_layer_scales, act_dict


# class SmoothLlamaForCausalLM(BaseSmoothForCausalLM):
#     def __init__(self, model: PreTrainedModel, quantized: bool = False):
#         super().__init__(model, quantized)

#     def quantized(
#         self,
#         tokenizer,
#         dataset_path,
#         num_samples=512,
#         seq_len=512,
#     ):
#         llama_model = self.model
#         llama_config = llama_model.config

#         llama_model.eval()
#         device = next(llama_model.parameters()).device
#         # print("model:", llama_model)
#         act_dict = defaultdict(dict)

#         def stat_io_hook(m, x, y, name):
#             if isinstance(x, tuple):
#                 x = x[0]
#             if name not in act_dict or "input" not in act_dict[name]:
#                 act_dict[name]["input"] = x.detach().abs().max().item()
#             else:
#                 act_dict[name]["input"] = max(act_dict[name]["input"], x.detach().abs().max().item())
#             if isinstance(y, tuple):
#                 y = y[0]
#             if name not in act_dict or "output" not in act_dict[name]:
#                 act_dict[name]["output"] = y.detach().abs().max().item()
#             else:
#                 act_dict[name]["output"] = max(act_dict[name]["output"], y.detach().abs().max().item())

#         for name, m in llama_model.named_modules():
#             if isinstance(m, LlamaAttention):
#                 setattr(m, "q_apply_rotary", LlamaApplyRotary())
#                 setattr(m, "k_apply_rotary", LlamaApplyRotary())
#                 m.forward = types.MethodType(llama_decoder_layer_forward, m)

#         hooks = []
#         for name, m in llama_model.named_modules():
#             if isinstance(m, LlamaApplyRotary):
#                 hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))
#             if isinstance(m, torch.nn.Linear):
#                 hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

#         print("Collecting activation scales...")
#         pbar = tqdm(range(num_samples))
#         dataset = load_dataset("json", data_files=dataset_path, split="train")
#         dataset = dataset.shuffle(seed=42)
#         for i in pbar:
#             input_ids = tokenizer(
#                 dataset["rows"][0][i]["row"]["text"],
#                 return_tensors="pt",
#                 max_length=seq_len,
#                 truncation=True,
#             ).input_ids.to(device)
#             llama_model(input_ids)
#             mean_scale = np.mean([v["input"] for v in act_dict.values()])
#             pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
#         for hook in hooks:
#             hook.remove()

#         decoder_layer_scales = []

#         for idx in range(llama_config.num_hidden_layers):
#             scale_dict = {}
#             scale_dict["attn_input_scale"] = act_dict[f"model.layers.{idx}.self_attn.q_proj"]["input"] / 127
#             scale_dict["q_output_scale"] = act_dict[f"model.layers.{idx}.self_attn.q_proj"]["output"] / 127
#             scale_dict["k_output_scale"] = act_dict[f"model.layers.{idx}.self_attn.k_proj"]["output"] / 127
#             scale_dict["v_output_scale"] = act_dict[f"model.layers.{idx}.self_attn.v_proj"]["output"] / 127

#             scale_dict["q_rotary_output_scale"] = (
#                 act_dict[f"model.layers.{idx}.self_attn.q_apply_rotary"]["output"] / 127
#             )

#             scale_dict["k_rotary_output_scale"] = (
#                 act_dict[f"model.layers.{idx}.self_attn.k_apply_rotary"]["output"] / 127
#             )

#             scale_dict["out_input_scale"] = act_dict[f"model.layers.{idx}.self_attn.o_proj"]["input"] / 127
#             # mlp scales
#             scale_dict["gate_input_scale"] = act_dict[f"model.layers.{idx}.mlp.gate_proj"]["input"] / 127
#             scale_dict["up_input_scale"] = act_dict[f"model.layers.{idx}.mlp.up_proj"]["input"] / 127
#             scale_dict["down_input_scale"] = act_dict[f"model.layers.{idx}.mlp.down_proj"]["input"] / 127

#             decoder_layer_scales.append(scale_dict)

#         for i, layer in enumerate(llama_model.model.layers):
#             orig_layer = layer
#             llama_model.model.layers[i] = LlamaSmoothquantDecoderLayer.pack(orig_layer, **decoder_layer_scales[i])

#         llama_model.model.forward = types.MethodType(llama_model_forward, llama_model.model)

#         cos, sin = init_to_get_rotary(llama_config)
#         llama_model.model.register_buffer("_cos_cached", cos)
#         llama_model.model.register_buffer("_sin_cached", sin)

#     def make_smooth_model(cls, llama_model):
#         super().make_smooth_model()

#         llama_config = llama_model.config

#         for i, layer in enumerate(llama_model.model.layers):
#             llama_model.model.layers[i] = LlamaSmoothquantDecoderLayer(llama_config)

#         llama_model.model.forward = types.MethodType(llama_model_forward, llama_model.model)
#         cos, sin = init_to_get_rotary(llama_config)
#         llama_model.model.register_buffer("_cos_cached", cos)
#         llama_model.model.register_buffer("_sin_cached", sin)
