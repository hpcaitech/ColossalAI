# Code modified from smoothquant: https://github.com/mit-han-lab/smoothquant

from typing import Optional, Tuple

import torch
from torch import nn
from torch_int.nn.bmm import BMM_S8T_S8N_F32T, BMM_S8T_S8N_S8T
from torch_int.nn.linear import W8A8B8O8Linear, W8A8BFP32OFP32Linear
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

from colossalai.kernel.triton import int8_rotary_embedding_fwd

from .linear import W8A8BFP32O32LinearSiLU


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

        self.attention_weight_scale = 1.0

        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        self.k_proj = W8A8B8O8Linear(hidden_size, hidden_size)
        self.v_proj = W8A8B8O8Linear(hidden_size, hidden_size)
        self.q_proj = W8A8B8O8Linear(hidden_size, hidden_size)
        self.out_proj = W8A8BFP32OFP32Linear(hidden_size, hidden_size)

        self.q_output_scale = torch.tensor([1.0])
        self.k_output_scale = torch.tensor([1.0])
        self.rotary_output_scale = torch.tensor([1.0])

    def pack(
        self,
        module: LlamaAttention,
        input_scale: float,
        q_output_scale: float,
        k_output_scale: float,
        v_output_scale: float,
        out_input_scale: float,
        rotary_output_scale: float,
    ):
        int8_module = LLamaSmoothquantAttention(module.hidden_size, module.head_dim)
        int8_module.q_output_scale = q_output_scale
        int8_module.k_output_scale = k_output_scale
        int8_module.rotary_output_scale = rotary_output_scale
        q_output_scale = q_output_scale * module.scaling
        module.q_proj.weight *= module.scaling
        module.q_proj.bias *= module.scaling
        int8_module.q_proj = W8A8B8O8Linear.from_float(module.q_proj, input_scale, q_output_scale)

        int8_module.k_proj = W8A8B8O8Linear.from_float(module.k_proj, input_scale, k_output_scale)
        int8_module.v_proj = W8A8B8O8Linear.from_float(module.v_proj, input_scale, v_output_scale)
        int8_module.out_proj = W8A8BFP32OFP32Linear.from_float(module.out_proj, out_input_scale)
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(q_output_scale, k_output_scale)

        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        int8_module.pv_bmm = BMM_S8T_S8N_S8T.from_scale(1.0 / 127, v_output_scale, out_input_scale)
        return int8_module

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor],
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.size()
        # get query proj
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        cos = rotary_emb[0]
        sin = rotary_emb[1]
        int8_rotary_embedding_fwd(
            query_states.view(-1, self.num_heads, self.head_dim),
            cos,
            sin,
            self.q_output_scale,
            self.rotary_output_scale,
        )
        int8_rotary_embedding_fwd(
            key_states.view(-1, self.num_heads, self.head_dim),
            cos,
            sin,
            self.k_output_scale,
            self.rotary_output_scale,
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)

        past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)

        query_states = self._shape(query_states, seq_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = self.qk_bmm(query_states, key_states)

        if attn_weights.size() != (bsz * self.num_heads, seq_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, seq_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, seq_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, seq_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, seq_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, seq_len, src_len)

        attn_probs = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_probs = layer_head_mask.view(1, -1, 1, 1) * attn_probs.view(bsz, self.num_heads, seq_len, src_len)
            attn_probs = attn_probs.view(bsz * self.num_heads, seq_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_probs_reshaped = attn_probs.view(bsz, self.num_heads, seq_len, src_len)
            attn_probs = attn_probs_reshaped.view(bsz * self.num_heads, seq_len, src_len)
        else:
            attn_probs_reshaped = None

        # (A_row V_row)_row = (A_row V_col ^T)_row
        attn_probs.mul_(127).round_()
        attn_probs = attn_probs.to(torch.int8)

        value_states = value_states.transpose(1, 2).contiguous()
        attn_output = self.pv_bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, seq_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_probs_reshaped, past_key_value


class LlamaSmoothquantMLP(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.gate_proj = W8A8BFP32O32LinearSiLU(hidden_size, intermediate_size)
        self.up_proj = W8A8BFP32OFP32Linear(hidden_size, intermediate_size)
        self.down_proj = W8A8BFP32OFP32Linear(intermediate_size, hidden_size)
        self.down_proj_input_scale = 1.0
        self.inter_out_scale = 1.0

    def pack(
        self,
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
        self.down_proj_input_scale = down_proj_input_scale
        return int8_module

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        x_shape = hidden_states.shape
        gate_out = self.gate_proj(hidden_states)
        up_out = self.up_proj(hidden_states)
        inter_out = gate_out * up_out
        inter_out = inter_out.div_(self.inter_out_scale).round().clamp(-128, 127).to(torch.int8)
        down_out = self.down_proj(inter_out)
        down_out = down_out.view(*x_shape[:-1], -1)
        return down_out
