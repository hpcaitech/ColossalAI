# This is modified from huggingface transformers
# https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/models/llama/modeling_llama.py
import warnings
from types import MethodType
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaForCausalLM,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from colossalai.inference.spec import GlideInput
from colossalai.kernel.triton import flash_decoding_attention
from colossalai.logging import get_dist_logger

logger = get_dist_logger(__name__)


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


def glide_llama_causal_lm_forward(
    self: LlamaForCausalLM,
    input_ids: torch.LongTensor = None,
    glide_input: Optional[GlideInput] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
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
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        glide_input=glide_input,
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
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    if not return_dict:
        output = (logits,) + outputs[1:]
        return output

    return CausalLMOutputWithPast(
        loss=None,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def glide_llama_model_forward(
    self: LlamaModel,
    input_ids: torch.LongTensor = None,
    glide_input: GlideInput = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_seen_tokens = 0
    if use_cache:  # kept for BC (cache positions)
        if not isinstance(past_key_values, StaticCache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

    if cache_position is None:
        if isinstance(past_key_values, StaticCache):
            raise ValueError("cache_position is a required argument when using StaticCache.")
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    attention_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # GlideLlamaDecoderLayer
        layer_outputs = decoder_layer(
            hidden_states,
            glide_input=glide_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
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
        next_cache = (
            next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
        )
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.large_hidden_size = large_hidden_size
        self.large_num_attention_heads = large_num_attention_heads


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
        glide_input: GlideInput = None,  # Used for glimpsing main model's KV caches
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Optional[torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()

        block_tables = glide_input.block_tables
        large_k_cache = glide_input.large_k_cache
        large_v_cache = glide_input.large_v_cache
        sequence_lengths = glide_input.sequence_lengths
        cache_block_size = large_k_cache.size(-2)

        query_states = self.q_proj(hidden_states)
        kv_seq_len = sequence_lengths.max().item()

        query_states = query_states.view(bsz, -1, self.large_num_heads, self.large_head_dim).transpose(1, 2)

        # for RoPE
        position_ids = position_ids + glide_input.n_spec_tokens
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states = apply_single_rotary_pos_emb(query_states, cos, sin, position_ids)
        query_states = query_states.transpose(1, 2)
        query_states = query_states.reshape(-1, self.large_num_heads, self.large_head_dim)

        attn_output = flash_decoding_attention(
            q=query_states,
            k_cache=large_k_cache,
            v_cache=large_v_cache,
            kv_seq_len=sequence_lengths,
            block_tables=block_tables,
            block_size=cache_block_size,
            max_seq_len_in_batch=kv_seq_len,
        )  # attn_output: [bsz * q_len, num_heads * head_dim]

        attn_output = attn_output.reshape(bsz, q_len, self.large_hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output


# A class to be used to replace LlamaDecoderLayer in a Llama Model as Drafter in speculative decoding.
# Refer to GLIDE with a CAPE https://arxiv.org/pdf/2402.02082.pdf
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
        glide_input: GlideInput = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
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

        curr_q_len = hidden_states.size(1)
        # Cross attention
        if glide_input is None or not glide_input.glimpse_ready:
            warnings.warn(
                "Data used for glimpsing the past KV caches of the main model (verifier) is not complete. "
                "Fall back to normal decoder layer modeling (drafter). "
                "This might lead to incorrect results when using the Glide Models for speculative decoding."
            )
        elif curr_q_len == 1:
            # Notice that we skip prefill stage
            # always use the output of the main model as the inputs for the next round of speculation
            residual = hidden_states

            hidden_states = self.cross_attn(
                hidden_states=hidden_states,
                glide_input=glide_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
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

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class GlideLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: GlideLlamaConfig):
        super().__init__(config)
        self.config = config
        bound_method = MethodType(glide_llama_causal_lm_forward, self)
        setattr(self, "forward", bound_method)
        bound_method = MethodType(glide_llama_model_forward, self.model)
        model = getattr(self, "model")
        setattr(model, "forward", bound_method)
        replaced_layers = nn.ModuleList(
            [GlideLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        setattr(model, "layers", replaced_layers)
