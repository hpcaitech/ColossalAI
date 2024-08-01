# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch OpenMoE model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRMSNorm
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

try:
    # TODO: remove this after updating openmoe example
    # NOTE(yuanheng-zhao): This is a temporary fix for the issue that
    # the flash_attention module is not imported correctly for different CI tests.
    # We replace the import path `colossalai.kernel.extensions.flash_attention`
    # because in the current example test, colossalai version <= 0.3.6 is installed,
    # where `colossalai.kernel.extensions.flash_attention` is still valid;
    # however in unit test `test_moe_checkpoint`, the lastest version of colossalai is installed,
    # where extension has been refactored and the path is not valid.
    import flash_attention  # noqa

    HAS_FLASH_ATTN = True
except:
    HAS_FLASH_ATTN = False
from colossalai.kernel.triton.llama_act_combine_kernel import HAS_TRITON
from colossalai.legacy.moe.manager import MOE_MANAGER
from colossalai.legacy.moe.utils import get_activation, set_moe_args
from colossalai.shardformer.layer.moe import SparseMLP

if HAS_TRITON:
    from colossalai.kernel.triton.llama_act_combine_kernel import LlamaActCombine

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def set_openmoe_args(
    config: LlamaConfig,
    num_experts: int,
    moe_layer_interval: int,
    router_topk: int = 2,
    router_capacity_factor_train: float = 1.25,
    router_capacity_factor_eval: float = 2.0,
    router_min_capacity: int = 4,
    router_noisy_policy: str = None,
    router_drop_tks: bool = True,
    router_aux_loss_factor: float = 0.01,
    router_z_loss_factor: float = 0.0001,
    mlp_gated: bool = True,
    label_smoothing: float = 0.001,
    z_loss_factor: float = 0.01,
    enable_load_balance: bool = False,
    load_balance_tolerance: float = 0.1,
    load_balance_beam_width: int = 8,
    load_balance_group_swap_factor: float = 0.4,
    enable_kernel: bool = False,
    enable_comm_overlap: bool = False,
    enable_hierarchical_alltoall: bool = True,
) -> None:
    """
    MoE related arguments.
    It inserts the MoE arguments into the Llama config.

    Args:
        config (LlamaConfig): Transformers Llama config.
        num_experts (int, optional): Number of experts.
        moe_layer_interval (int, optional): The interval moe layer.
        router_topk (int, optional): Moe router top k. Defaults to 2.
        router_capacity_factor_train (float, optional): Moe router max capacity for train. Defaults to 1.25.
        router_capacity_factor_eval (float, optional): Moe router max capacity for eval. Defaults to 2.0.
        router_min_capacity (int, optional): Moe router min capacity. Defaults to 4.
        router_noisy_policy (str, optional): Moe router noisy policy. You can choose [Jitter, Gaussian, None]. Defaults to None.
        router_drop_tks (bool, optional): Whether moe router drop tokens which exceed max capacity. Defaults to True.
        router_aux_loss_factor (float, optional): Moe router aux loss. You can refer to STMoE for details. Defaults to 0.01.
        router_z_loss_factor (float, optional): Moe router z loss. You can refer to STMoE for details. Defaults to 0.01.
        mlp_gated (bool, optional): Use gate in mlp. Defaults to True.
        label_smoothing (float, optional): Label smoothing. Defaults to 0.001.
        z_loss_factor (float, optional): The final outputs' classification z loss factor. Defaults to 0.01.
        enable_load_balance (bool, optional): Expert load balance. Defaults to False.
        load_balance_tolerance (float, optional): Expert load balance search's difference tolerance. Defaults to 0.1.
        load_balance_beam_width (int, optional): Expert load balance search's beam width. Defaults to 8.
        load_balance_group_swap_factor (float, optional): Expert load balance group swap factor. Longer value encourages less swap. Defaults to 0.4.
        enable_kernel (bool, optional): Use kernel optimization. Defaults to False.
        enable_comm_overlap (bool, optional): Use communication overlap for MoE. Recommended to enable for multi-node training. Defaults to False.
        enable_hierarchical_alltoall (bool, optional): Use hierarchical alltoall for MoE. Defaults to False.
    """
    moe_args = dict(
        num_experts=num_experts,
        moe_layer_interval=moe_layer_interval,
        router_topk=router_topk,
        router_capacity_factor_train=router_capacity_factor_train,
        router_capacity_factor_eval=router_capacity_factor_eval,
        router_min_capacity=router_min_capacity,
        router_noisy_policy=router_noisy_policy,
        router_drop_tks=router_drop_tks,
        router_aux_loss_factor=router_aux_loss_factor,
        router_z_loss_factor=router_z_loss_factor,
        mlp_gated=mlp_gated,
        label_smoothing=label_smoothing,
        z_loss_factor=z_loss_factor,
        enable_load_balance=enable_load_balance,
        load_balance_tolerance=load_balance_tolerance,
        load_balance_beam_width=load_balance_beam_width,
        load_balance_group_swap_factor=load_balance_group_swap_factor,
        enable_kernel=enable_kernel,
        enable_comm_overlap=enable_comm_overlap,
        enable_hierarchical_alltoall=enable_hierarchical_alltoall,
    )
    set_moe_args(config, moe_args)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def generate_fixed_pos_embedding(features, length, min_timescale=1.0, max_timescale=10000.0):
    """Generate Sin/Cos for Rotary Embeddings.

    Args:
      features: an integer
      length: an integer
      min_timescale: an optional float
      max_timescale: an optional float

    Returns:
      output_sin: a float32 Tensor with shape [length, features]
      output_cos: a float32 Tensor with shape [length, features]
    """
    fraction = torch.arange(0, features, 2, dtype=torch.float32).cuda() / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    rotational_frequency = 1.0 / timescale

    sinusoid_inp = torch.einsum("i,j->ij", torch.arange(length, dtype=torch.float32).cuda(), rotational_frequency)

    sinusoid_inp = torch.cat([sinusoid_inp, sinusoid_inp], dim=-1)

    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def apply_rotary_embedding(q, k, cos, sin, decode=False, rotary_index=None):
    """Helper function to apply Rotary Embeddings."""
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)

    if len(k.shape) == 3:
        # for multi query attention
        k = k.unsqueeze(2)
        multiquery = True
    else:
        multiquery = False

    batch, qlen, qheads, d = q.shape
    kbatch, klen, kheads, kd = k.shape
    assert batch == kbatch, f"{batch} != {kbatch}"
    assert d == kd, f"{d} != {kd}"
    if decode and qlen == 1 and rotary_index is not None:
        qcos = cos[rotary_index + 1, :]
        qsin = sin[rotary_index + 1, :]
        qcos = qcos.unsqueeze(2)
        qsin = qsin.unsqueeze(2)
        kcos, ksin = cos[:klen, :], sin[:klen, :]
        kcos = kcos.unsqueeze(0).unsqueeze(2)
        ksin = ksin.unsqueeze(0).unsqueeze(2)
    else:
        qcos, qsin = cos[:qlen, :], sin[:qlen, :]
        qcos = qcos.unsqueeze(0).unsqueeze(2)
        qsin = qsin.unsqueeze(0).unsqueeze(2)
        kcos, ksin = qcos, qsin

    out_q = (q * qcos) + (rotate_half(q) * qsin)
    out_k = (k * kcos) + (rotate_half(k) * ksin)

    if multiquery:
        out_k = out_k.squeeze(2)

    return out_q, out_k


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def SwiGLU(x):
    """Gated linear unit activation function.
    Args:
        x : input array
        axis: the axis along which the split should be computed (default: -1)
    """
    size = x.shape[-1]
    assert size % 2 == 0, "axis size must be divisible by 2"
    x1, x2 = torch.split(x, size // 2, -1)
    return x1 * (x2 * torch.sigmoid(x2))


class OpenMoeMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.hidden_act = config.hidden_act
        self.act_fn = get_activation(self.hidden_act)
        self.use_kernel = config.enable_kernel

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            if HAS_TRITON and self.use_kernel and self.hidden_act == "swiglu":
                down_proj = self.down_proj(LlamaActCombine.apply(self.gate_proj(x), self.up_proj(x)))
            else:
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


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


class OpenMoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.sin, self.cos = generate_fixed_pos_embedding(self.head_dim, self.max_position_embeddings, 1.0, 1e4)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_kernel: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
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
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        max_length = max(query_states.shape[1], key_states.shape[1])
        assert max_length <= self.sin.shape[0]
        sin, cos = self.sin[:max_length], self.cos[:max_length]
        # TODO: for inference, we can add emb kv into cache to avoid computation
        query_states, key_states = apply_rotary_embedding(
            query_states, key_states, cos, sin, decode=True if q_len == 1 else False, rotary_index=position_ids
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if HAS_FLASH_ATTN and use_kernel:
            from flash_attn import flash_attn_func

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            attn_output = flash_attn_func(query_states, key_states, value_states, softmax_scale=1.0, causal=True)
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
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
                if self.training:
                    attention_mask = attention_mask.clone().detach()
                attention_mask[:, :, :, 0] = 0
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
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class OpenMoeDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, moe: bool):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.moe = moe
        self.self_attn = OpenMoeAttention(config=config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.moe:
            self.mlp = SparseMLP(
                num_experts=config.num_experts,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                router_top_k=config.router_topk,
                router_capacity_factor_train=config.router_capacity_factor_train,
                router_capacity_factor_eval=config.router_capacity_factor_eval,
                router_min_capacity=config.router_min_capacity,
                router_noisy_policy=config.router_noisy_policy,
                router_drop_tks=config.router_drop_tks,
                mlp_activation=config.hidden_act,
                mlp_gated=config.mlp_gated,
                enable_load_balance=config.enable_load_balance,
                load_balance_tolerance=config.load_balance_tolerance,
                load_balance_beam_width=config.load_balance_beam_width,
                load_balance_group_swap_factor=config.load_balance_group_swap_factor,
                enable_kernel=config.enable_kernel,
                enable_hierarchical_comm=config.enable_hierarchical_alltoall,
            )
            self.pre_extra_mlp_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.extra_mlp = OpenMoeMLP(config)
        else:
            self.mlp = OpenMoeMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
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
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.moe:
            residual = hidden_states
            hidden_states = self.pre_extra_mlp_layernorm(hidden_states)
            hidden_states = self.extra_mlp(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class OpenMoePreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, OpenMoeModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class OpenMoeModel(OpenMoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                OpenMoeDecoderLayer(config, moe=True if (i + 1) % config.moe_layer_interval == 0 else False)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
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
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

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
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
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
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
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


class OpenMoeForCausalLM(OpenMoePreTrainedModel):
    # _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OpenMoeModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        chunk_head: Optional[bool] = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        # reset moe loss
        MOE_MANAGER.reset_loss()  # TODO: remove

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)

        loss = None
        # if no training, just do forward
        if labels is None:
            logits = self.lm_head(hidden_states)
            logits = logits.float()
        # the vocab size for openmoe is 30w+
        # which causes great activation memory in training, up to 20G for one sequence
        # so we use chunk and checkpoint to reduce memory
        else:
            if chunk_head == True:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        logits = module(inputs[0])
                        logits = logits.float()
                        # Shift so that tokens < n predict n
                        shift_logits = logits[..., :-1, :].contiguous().float()
                        shift_labels = inputs[1][..., 1:].contiguous()
                        # Flatten the tokens
                        loss = self._calculate_loss(shift_logits, shift_labels)
                        return loss

                    return custom_forward

                aux_loss, z_loss = self._calculate_router_loss()
                loss = aux_loss + z_loss
                for batch_idx in range(hidden_states.shape[0]):
                    loss = loss + torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.lm_head),
                        hidden_states[batch_idx : batch_idx + 1, :],
                        labels[batch_idx : batch_idx + 1, :],
                    )
                logits = None
            else:
                logits = self.lm_head(hidden_states)
                logits = logits.float()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                aux_loss, z_loss = self._calculate_router_loss()
                loss = aux_loss + z_loss
                loss = loss + self._calculate_loss(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def _calculate_router_loss(self, aux_loss: list = None, z_loss: list = None):
        if aux_loss is None or z_loss is None:
            aux_loss, z_loss = MOE_MANAGER.get_loss()  # TODO: remove
        assert len(aux_loss) == len(z_loss) == self.config.num_hidden_layers // self.config.moe_layer_interval
        aux_loss = self.config.router_aux_loss_factor * sum(aux_loss) / len(aux_loss)
        z_loss = self.config.router_z_loss_factor * sum(z_loss) / len(z_loss)
        return aux_loss, z_loss

    def _calculate_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy and entropy for log probs and targets.

        Args:
            logits: [batch, length, num_classes] float array.
            targets: categorical targets [batch, length] int array.

        Returns:
            Tuple of scalar loss.
        """
        if len(logits.shape) != len(targets.shape) + 1:
            raise ValueError(
                "Incorrect shapes. Got shape %s logits and %s targets" % (str(logits.shape), str(targets.shape))
            )
        vocab_size = logits.shape[-1]
        confidence = 1.0 - self.config.label_smoothing
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(
            confidence * math.log(confidence) + (vocab_size - 1) * low_confidence * math.log(low_confidence + 1e-20)
        )

        # one hot
        soft_targets = targets[..., None] == torch.arange(vocab_size, device=targets.device).reshape(
            (1,) * len(targets.shape) + (-1,)
        )
        soft_targets = torch.where(
            soft_targets, torch.full_like(soft_targets, confidence), torch.full_like(soft_targets, low_confidence)
        )
        soft_targets = soft_targets.to(torch.float32)

        # cross entropy
        total_loss = ZLossCrossEntropy.apply(logits, soft_targets, self.config.z_loss_factor)
        total_loss = total_loss - normalizing_constant
        total_loss = torch.mean(torch.sum(total_loss, dim=-1), dim=0)
        return total_loss


class ZLossCrossEntropy(torch.autograd.Function):
    """Computes cross entropy loss with stable custom gradient.

    Computes a stabilized-gradient version of:
        -jnp.sum(targets * nn.log_softmax(logits), axis=-1)

    If z_loss > 0, then an auxiliary loss equal to z_loss*log(z)^2
    will be added to the cross entropy loss (z = softmax normalization constant).
    The two uses of z_loss are:
    1. To keep the logits from drifting too far from zero, which can cause
        unacceptable roundoff errors in bfloat16.
    2. To encourage the logits to be normalized log-probabilities.

    Args:
        logits: [batch, length, num_classes] float array.
        targets: categorical one-hot targets [batch, length, num_classes] float
        array.
        z_loss: coefficient for auxilliary z-loss loss term.

    Returns:
        tuple with the total loss and the z_loss, both
        float arrays with shape [batch, length].
    """

    @staticmethod
    def forward(ctx, logits, targets, z_loss):
        max_logit = torch.max(logits, dim=-1, keepdim=True)[0]
        shifted = logits - max_logit
        exp_shifted = torch.exp(shifted)
        sum_exp = torch.sum(exp_shifted, axis=-1, keepdims=True)
        sum_exp_log = torch.log(sum_exp)
        log_softmax = shifted - sum_exp_log
        loss = -torch.sum(targets * log_softmax, axis=-1)
        # Add auxilliary z-loss term.
        log_z = torch.squeeze(sum_exp_log + max_logit, axis=-1)
        total_z_loss = z_loss * torch.square(log_z)
        loss += total_z_loss
        ctx.z_loss = z_loss
        ctx.save_for_backward(logits, targets, exp_shifted, sum_exp, log_softmax, log_z)
        return loss

    @staticmethod
    def backward(ctx, *grad_outputs):
        assert len(grad_outputs) == 1
        g = grad_outputs[0]
        z_loss = ctx.z_loss
        logits, targets, exp_shifted, sum_exp, log_softmax, log_z = ctx.saved_tensors
        # z-loss term adds the (2 * z_loss * log_z) factor.
        deriv = (1 + 2 * z_loss * log_z).unsqueeze(-1) * exp_shifted / sum_exp - targets
        g_logits = g.unsqueeze(-1) * deriv
        g_targets = -g.unsqueeze(-1) * log_softmax

        return (
            g_logits.to(logits.dtype),
            g_targets.to(targets.dtype),
            None,
        )
