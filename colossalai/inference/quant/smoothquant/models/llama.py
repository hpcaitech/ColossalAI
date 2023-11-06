import math
import os
import types
from collections import defaultdict
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_int.nn.bmm import BMM_S8T_S8N_F32T, BMM_S8T_S8N_S8T
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LLAMA_INPUTS_DOCSTRING,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRotaryEmbedding,
    repeat_kv,
    rotate_half,
)
from transformers.utils import add_start_docstrings_to_model_forward

from colossalai.inference.tensor_parallel.batch_infer_state import BatchInferState
from colossalai.kernel.triton import (
    copy_kv_cache_to_dest,
    int8_rotary_embedding_fwd,
    smooth_llama_context_attn_fwd,
    smooth_token_attention_fwd,
)

from .base_model import BaseSmoothForCausalLM
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

        int8_module.attn_input_scale = torch.tensor([attn_input_scale])

        int8_module.q_output_scale = torch.tensor([q_output_scale])
        int8_module.k_output_scale = torch.tensor([k_output_scale])
        int8_module.v_output_scale = torch.tensor([v_output_scale])

        int8_module.q_rotary_output_scale = torch.tensor([q_rotary_output_scale])
        int8_module.k_rotary_output_scale = torch.tensor([k_rotary_output_scale])

        int8_module.q_proj = W8A8B8O8Linear.from_float(module.q_proj, attn_input_scale, q_output_scale)
        int8_module.k_proj = W8A8B8O8Linear.from_float(module.k_proj, attn_input_scale, k_output_scale)
        int8_module.v_proj = W8A8B8O8Linear.from_float(module.v_proj, attn_input_scale, v_output_scale)
        int8_module.o_proj = W8A8BFP32OFP32Linear.from_float(module.o_proj, out_input_scale)

        int8_module.out_input_scale = torch.tensor([out_input_scale])

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
        infer_state: Optional[BatchInferState] = None,
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

        def _copy_kv_to_mem_cache(layer_id, key_buffer, value_buffer, context_mem_index, mem_manager):
            copy_kv_cache_to_dest(key_buffer, context_mem_index, mem_manager.key_buffer[layer_id])
            copy_kv_cache_to_dest(value_buffer, context_mem_index, mem_manager.value_buffer[layer_id])
            return

        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_heads, self.head_dim)
        value_states = value_states.view(-1, self.num_heads, self.head_dim)

        if infer_state.is_context_stage:
            # first token generation

            # copy key and value calculated in current step to memory manager
            _copy_kv_to_mem_cache(
                infer_state.decode_layer_id,
                key_states,
                value_states,
                infer_state.context_mem_index,
                infer_state.cache_manager,
            )

            attn_output = torch.empty_like(query_states)

            smooth_llama_context_attn_fwd(
                query_states,
                key_states,
                value_states,
                attn_output,
                self.q_rotary_output_scale.item(),
                self.k_rotary_output_scale.item(),
                self.v_output_scale.item(),
                self.out_input_scale.item(),
                infer_state.start_loc,
                infer_state.seq_len,
                q_len,
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
                _copy_kv_to_mem_cache(
                    infer_state.decode_layer_id,
                    key_states,
                    value_states,
                    infer_state.decode_mem_index,
                    infer_state.cache_manager,
                )

            # (batch_size, seqlen, nheads, headdim)
            attn_output = torch.empty_like(query_states)

            smooth_token_attention_fwd(
                query_states,
                infer_state.cache_manager.key_buffer[infer_state.decode_layer_id],
                infer_state.cache_manager.value_buffer[infer_state.decode_layer_id],
                attn_output,
                self.q_rotary_output_scale.item(),
                self.k_rotary_output_scale.item(),
                self.v_output_scale.item(),
                self.out_input_scale.item(),
                infer_state.block_loc,
                infer_state.start_loc,
                infer_state.seq_len,
                infer_state.max_len_in_batch,
            )

        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, None


class LlamaLayerNormQ(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.input_scale = 1.0
        self.variance_epsilon = eps
        self.register_buffer("weight", torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        ln_output_fp = torch.nn.functional.layer_norm(x, x.shape[-1:], self.weight, None, self.variance_epsilon)
        ln_output_int8 = ln_output_fp.round().clamp(-128, 127).to(torch.int8)
        return ln_output_int8

    @staticmethod
    def from_float(module: torch.nn.LayerNorm, output_scale: float):
        assert module.weight.shape[0] == module.weight.numel()
        q_module = LlamaLayerNormQ(module.weight.shape[0], module.variance_epsilon)
        q_module.weight = module.weight / output_scale
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
        int8_module.down_proj_input_scale = torch.tensor([down_proj_input_scale])
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

        int8_decoder_layer.input_layernorm = LlamaLayerNormQ.from_float(module.input_layernorm, attn_input_scale)
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

        int8_decoder_layer.post_attention_layernorm = LlamaLayerNormQ.from_float(
            module.post_attention_layernorm, gate_input_scale
        )

        int8_decoder_layer.mlp = LlamaSmoothquantMLP.pack(
            module.mlp,
            gate_input_scale,
            up_input_scale,
            down_input_scale,
        )

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
        infer_state: Optional[BatchInferState] = None,
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
            infer_state=infer_state,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None, None


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


# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
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


# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
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

    infer_state = self.infer_state
    if infer_state.is_context_stage:
        past_key_values_length = 0
    else:
        past_key_values_length = infer_state.max_len_in_batch - 1

    seq_length_with_past = seq_length + past_key_values_length

    # NOTE: differentiate with prefill stage
    #       block_loc require different value-assigning method for two different stage
    # NOTE: differentiate with prefill stage
    #       block_loc require different value-assigning method for two different stage
    if infer_state.is_context_stage:
        infer_state.context_mem_index = infer_state.cache_manager.alloc(infer_state.total_token_num)
        infer_state.init_block_loc(
            infer_state.block_loc, infer_state.seq_len, seq_length, infer_state.context_mem_index
        )
    else:
        alloc_mem = infer_state.cache_manager.alloc_contiguous(batch_size)
        if alloc_mem is not None:
            infer_state.decode_is_contiguous = True
            infer_state.decode_mem_index = alloc_mem[0]
            infer_state.decode_mem_start = alloc_mem[1]
            infer_state.decode_mem_end = alloc_mem[2]
            infer_state.block_loc[:, seq_length_with_past - 1] = infer_state.decode_mem_index
        else:
            print(f" *** Encountered allocation non-contiguous")
            print(f"    infer_state.cache_manager.max_len_in_batch: {infer_state.max_len_in_batch}")
            infer_state.decode_is_contiguous = False
            alloc_mem = infer_state.cache_manager.alloc(batch_size)
            infer_state.decode_mem_index = alloc_mem
            infer_state.block_loc[:, seq_length_with_past - 1] = infer_state.decode_mem_index

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
        raise NotImplementedError("not implement gradient_checkpointing and training options ")

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
    infer_state.decode_layer_id = 0
    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        layer_outputs = decoder_layer(
            hidden_states,
            rotary_emb=(position_cos, position_sin),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            infer_state=infer_state,
        )

        hidden_states = layer_outputs[0]
        infer_state.decode_layer_id += 1

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    infer_state.is_context_stage = False
    infer_state.start_loc = infer_state.start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
    infer_state.seq_len += 1
    infer_state.max_len_in_batch += 1

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


class SmoothLlamaForCausalLM(BaseSmoothForCausalLM):
    layer_type = "LlamaDecoderLayer"

    def __init__(self, model: PreTrainedModel, quantized: bool = False):
        super().__init__(model, quantized)

    # Adatped from https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/calibration.py
    def get_act_dict(
        self,
        tokenizer,
        dataset,
        num_samples=512,
        seq_len=512,
    ):
        llama_model = self.model

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

        self.collect_act_dict(llama_model, tokenizer, dataset, act_dict, device, num_samples, seq_len)

        for hook in hooks:
            hook.remove()
        return act_dict

    def smooth_fn(self, scales, alpha=0.5):
        model = self.model
        for name, module in model.named_modules():
            if isinstance(module, LlamaDecoderLayer):
                attn_ln = module.input_layernorm
                qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
                qkv_input_scales = scales[name + ".self_attn.q_proj"]
                self.smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

    def create_quantized_model(model):
        llama_config = model.config
        for i, layer in enumerate(model.model.layers):
            model.model.layers[i] = LlamaSmoothquantDecoderLayer(llama_config)

        model.model.forward = types.MethodType(llama_model_forward, model.model)
        cos, sin = init_to_get_rotary(llama_config)
        model.model.register_buffer("_cos_cached", cos)
        model.model.register_buffer("_sin_cached", sin)

    def quantized(
        self,
        tokenizer,
        dataset,
        num_samples=512,
        seq_len=512,
        alpha=0.5,
    ):
        llama_model = self.model
        llama_config = llama_model.config

        act_scales = self.get_act_scales(llama_model, tokenizer, dataset, num_samples, seq_len)

        self.smooth_fn(act_scales, alpha)

        act_dict = self.get_act_dict(tokenizer, dataset, num_samples, seq_len)
        decoder_layer_scales = []

        for idx in range(llama_config.num_hidden_layers):
            scale_dict = {}
            scale_dict["attn_input_scale"] = act_dict[f"model.layers.{idx}.self_attn.q_proj"]["input"] / 127
            scale_dict["q_output_scale"] = act_dict[f"model.layers.{idx}.self_attn.q_proj"]["output"] / 127
            scale_dict["k_output_scale"] = act_dict[f"model.layers.{idx}.self_attn.k_proj"]["output"] / 127
            scale_dict["v_output_scale"] = act_dict[f"model.layers.{idx}.self_attn.v_proj"]["output"] / 127

            scale_dict["q_rotary_output_scale"] = (
                act_dict[f"model.layers.{idx}.self_attn.q_apply_rotary"]["output"] / 127
            )
            scale_dict["k_rotary_output_scale"] = (
                act_dict[f"model.layers.{idx}.self_attn.k_apply_rotary"]["output"] / 127
            )

            scale_dict["out_input_scale"] = act_dict[f"model.layers.{idx}.self_attn.o_proj"]["input"] / 127

            scale_dict["gate_input_scale"] = act_dict[f"model.layers.{idx}.mlp.gate_proj"]["input"] / 127
            scale_dict["up_input_scale"] = act_dict[f"model.layers.{idx}.mlp.up_proj"]["input"] / 127
            scale_dict["down_input_scale"] = act_dict[f"model.layers.{idx}.mlp.down_proj"]["input"] / 127

            decoder_layer_scales.append(scale_dict)

        for i, layer in enumerate(llama_model.model.layers):
            orig_layer = layer
            llama_model.model.layers[i] = LlamaSmoothquantDecoderLayer.pack(orig_layer, **decoder_layer_scales[i])

        llama_model.model.forward = types.MethodType(llama_model_forward, llama_model.model)

        cos, sin = init_to_get_rotary(llama_config)
        llama_model.model.register_buffer("_cos_cached", cos.to(self.model.device))
        llama_model.model.register_buffer("_sin_cached", sin.to(self.model.device))
