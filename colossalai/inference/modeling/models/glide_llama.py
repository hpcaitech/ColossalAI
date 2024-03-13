import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    repeat_kv,
)

from colossalai.logging import get_dist_logger

logger = get_dist_logger(__name__)


# The following functions are adapted from huggingface transformers
# https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_single_rotary_pos_emb(q, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


# This code is adapted from huggingface transformers:
# https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/models/llama/modeling_llama.py
def glide_llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    large_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
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
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # GlideLlamaDecoderLayer
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            large_key_values=large_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


class GlideLlamaConfig(LlamaConfig):
    """Configuration class with specific arguments used by GLIDE llama model as a drafter"""

    def __init__(
        self,
        large_hidden_size=4096,
        large_num_attention_heads=32,
        use_remap=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_remap = use_remap
        self.large_hidden_size = large_hidden_size
        self.large_num_attention_heads = large_num_attention_heads


# This code is adapted from huggingface transformers
# https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/models/llama/modeling_llama.py
class LlamaCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GlideLlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # large model (verifier) configs
        self.large_hidden_size = config.large_hidden_size
        self.large_num_heads = config.large_num_attention_heads
        self.large_head_dim = self.large_hidden_size // self.large_num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.large_num_heads * self.large_head_dim, bias=False)
        self.o_proj = nn.Linear(self.large_num_heads * self.large_head_dim, self.hidden_size, bias=False)
        self._init_rope()
        self.use_remap = config.use_remap
        if self.use_remap:
            self.remap_k = nn.Linear(self.large_head_dim, self.large_head_dim)
            self.remap_v = nn.Linear(self.large_head_dim, self.large_head_dim)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.large_head_dim,
                max_position_embeddings=self.max_position_embeddings,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.large_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.large_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        large_key_values: Optional[Tuple[torch.Tensor]] = None,  # 2 x (bsz, num_heads, kv_len, head_dim)
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()
        large_bsz, large_head_num, kv_len, large_head_dim = large_key_values[0].size()
        query_states = self.q_proj(hidden_states)
        kv_seq_len = large_key_values[0].shape[-2]

        if self.use_remap:
            key_states = self.remap_k(large_key_values[0])
            value_states = self.remap_v(large_key_values[1])
        else:
            key_states, value_states = large_key_values

        query_states = query_states.view(bsz, -1, self.large_num_heads, self.large_head_dim).transpose(1, 2)

        # for RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len + 32)  # XXX 32 <- maximum n tokens speculate(?)
        query_states = apply_single_rotary_pos_emb(query_states, cos, sin, position_ids)
        if self.use_remap:
            key_states = apply_single_rotary_pos_emb(
                key_states, cos, sin, torch.arange(kv_seq_len).view(1, -1).to(key_states.device)
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.large_head_dim)

        if attn_weights.size() != (bsz, self.large_num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.large_num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        log_attention = None
        if output_attentions:
            log_attention = nn.functional.log_softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.large_num_heads, q_len, self.large_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.large_num_heads, q_len, self.large_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.large_hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, log_attention


# A class to be used to replace LlamaDecoderLayer in a Llama Model as Drafter in speculative decoding.
# Refer to GLIDE with a CAPE https://arxiv.org/pdf/2402.02082.pdf
# This code is adapted from huggingface transformers
# https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/models/llama/modeling_llama.py
class GlideLlamaDecoderLayer(nn.Module):
    def __init__(self, config: GlideLlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.cross_attn = LlamaCrossAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def from_native_module(module: LlamaDecoderLayer, *args, **kwargs) -> "GlideLlamaDecoderLayer":
        """Build a GlideLlamaDecoderLayer from a native LlamaDecoderLayer"""
        config: LlamaConfig = module.mlp.config  # XXX
        layer_idx = module.self_attn.layer_idx
        glide_config = GlideLlamaConfig(**config.to_dict())
        glide_decoder_layer = GlideLlamaDecoderLayer(glide_config, layer_idx=layer_idx)

        return glide_decoder_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        large_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

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
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Cross attention
        if large_key_values is not None:
            # skip if large key values are not provided
            residual = hidden_states

            hidden_states, cross_attn_weights, _ = self.cross_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                large_key_values=large_key_values,
                output_attentions=output_attentions,
                use_cache=True,
            )
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (cross_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
