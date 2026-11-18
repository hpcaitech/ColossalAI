import logging
from typing import Any, Callable, Optional, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import check_model_inputs
from .configuration_ouro import OuroConfig


logger = logging.getLogger(__name__)


def needs_universal_cache(
    cache: Optional[Cache], max_cache_size: Optional[int]
) -> bool:
    if cache is None:
        return True
    if isinstance(cache, UniversalTransformerCache):
        return False
    if not isinstance(cache, Cache):
        return False
    can_grow = getattr(cache, "layer_class_to_replicate", None) is not None
    if can_grow:
        # Dynamic caches can extend to any index, so let them be
        return False
    cache_layers = getattr(cache, "layers", [])
    if max_cache_size is not None and len(cache_layers) < max_cache_size:
        try:
            cached_tokens = cache.get_seq_length()
        except Exception:
            cached_tokens = 0
        if cached_tokens > 0:
            raise ValueError(
                "The provided cache cannot store all Universal Transformer iterations. Please "
                "instantiate Ouro.modeling_ouro.UniversalTransformerCache and pass it as past_key_values."
            )
        return True
    return False


class OuroMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class UniversalTransformerCache(Cache):
    """Cache implementation that supports Ouro's multi-step Universal Transformer loops."""

    def __init__(self, max_cache_size: Optional[int] = None):
        # We intentionally don't call super().__init__ because the parent assumes static cache sizes.
        self.key_cache: list[Optional[torch.Tensor]] = []
        self.value_cache: list[Optional[torch.Tensor]] = []
        self.layers: list[Any] = []  # attribute expected by HF Cache utilities
        self._seen_tokens = 0
        self.max_cache_size = max_cache_size

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")

        if self.max_cache_size is not None and layer_idx >= self.max_cache_size:
            raise IndexError(
                f"Cache index {layer_idx} exceeds configured max_cache_size={self.max_cache_size}. "
                "Check total_ut_steps and num_hidden_layers."
            )

        # Expand cache storage so the requested index is available.
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        cached_key = self.key_cache[layer_idx]
        cached_value = self.value_cache[layer_idx]

        if cached_key is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            if (
                key_states.shape[0] != cached_key.shape[0]
                or key_states.shape[1] != cached_key.shape[1]
                or key_states.shape[3] != cached_key.shape[3]
            ):
                raise ValueError(
                    "Cached and incoming key/value tensors must match on batch, head, and head_dim dimensions."
                )
            assert cached_value is not None
            self.key_cache[layer_idx] = torch.cat([cached_key, key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([cached_value, value_states], dim=2)

        result_key = self.key_cache[layer_idx]
        result_value = self.value_cache[layer_idx]
        assert result_key is not None and result_value is not None

        # Track sequence length using the first populated cache entry.
        self._seen_tokens = result_key.shape[2]
        return result_key, result_value

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if layer_idx < 0 or len(self.key_cache) <= layer_idx:
            return 0
        cached = self.key_cache[layer_idx]
        if cached is None:
            return 0
        return cached.shape[2]

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for idx, (key_entry, value_entry) in enumerate(
            zip(self.key_cache, self.value_cache)
        ):
            if key_entry is None:
                continue
            assert value_entry is not None
            device = key_entry.device
            self.key_cache[idx] = key_entry.index_select(0, beam_idx.to(device))
            self.value_cache[idx] = value_entry.index_select(0, beam_idx.to(device))

    @property
    def is_compileable(self) -> bool:
        return False

    def clear(self) -> None:
        logger.debug("Clearing UniversalTransformerCache")
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class OuroAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: OuroConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        current_ut: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                current_ut * self.config.num_hidden_layers + self.layer_idx,
                cache_kwargs,
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@use_kernel_forward_from_hub("RMSNorm")
class OuroRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        OuroRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class OuroDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OuroConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = OuroAttention(config=config, layer_idx=layer_idx)

        self.mlp = OuroMLP(config)
        self.input_layernorm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = OuroRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = OuroRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_2 = OuroRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class OuroPreTrainedModel(PreTrainedModel):
    config: OuroConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OuroDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": OuroDecoderLayer,
        "attentions": OuroAttention,
    }


class OuroRotaryEmbedding(nn.Module):
    def __init__(self, config: OuroConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class OuroModel(OuroPreTrainedModel):
    def __init__(self, config: OuroConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                OuroDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OuroRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.total_ut_steps = getattr(self.config, "total_ut_steps", 4)
        self.early_exit_gate = nn.Linear(config.hidden_size, 1)
        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = self.config.use_cache

        max_cache_size: Optional[int] = None
        if use_cache:
            total_ut_steps = getattr(self.config, "total_ut_steps", 1) or 1
            total_layers = getattr(self.config, "num_hidden_layers", None)
            if total_layers is not None:
                max_cache_size = total_layers * total_ut_steps

            if needs_universal_cache(past_key_values, max_cache_size):
                past_key_values = UniversalTransformerCache(max_cache_size)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states_list = []
        gate_list = []

        for current_ut in range(self.total_ut_steps):
            for decoder_layer in self.layers[: self.config.num_hidden_layers]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    current_ut=current_ut,
                    **kwargs,
                )

            hidden_states = self.norm(hidden_states)
            hidden_states_list.append(hidden_states)
            gate_list.append(self.early_exit_gate(hidden_states))

        return (
            BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            ),
            hidden_states_list,
            gate_list,
        )


@auto_docstring
class OuroForCausalLM(OuroPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = OuroModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 分块大小配置
        self.chunk_size = getattr(config, "chunk_size", 2)  # 默认分块大小为2
        self.early_exit_step = getattr(config, "early_exit_step", None)
        self.early_exit_threshold = getattr(config, "early_exit_threshold", None)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        use_weighted_exit: Optional[bool] = False,  # 控制是否使用加权 early exit
        exit_at_step: Optional[int] = None,
        exit_threshold: Optional[float] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Args:
            use_weighted_exit (`bool`, *optional*, defaults to `False`):
                Whether to use weighted early exit. If `True`, the logits from all UT steps will be
                averaged according to the exit probability distribution.
            exit_at_step (`int`, *optional*):
                Specifies which UT step to exit at. If set, the model will directly use the hidden states
                from this step to generate logits, ignoring other exit strategies.
            exit_threshold (`float`, *optional*):
                The cumulative probability threshold for early exit. When the cumulative exit probability
                reaches this threshold, the model will exit at that step.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OuroForCausalLM

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        exit_at_step = (
            exit_at_step if exit_at_step is not None else self.early_exit_step
        )
        exit_threshold = (
            exit_threshold if exit_threshold is not None else self.early_exit_threshold
        )

        outputs, hidden_states_list, gate_list = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        def _select_token_positions(tensor: torch.Tensor) -> torch.Tensor:
            if isinstance(slice_indices, slice):
                return tensor[:, slice_indices, ...]
            if isinstance(slice_indices, torch.Tensor):
                return tensor.index_select(1, slice_indices.to(tensor.device))
            raise TypeError(
                f"Unsupported index type for logits_to_keep: {type(slice_indices)}"
            )

        stacked_exit_pdf = None
        if gate_list:
            pdf_list = []
            remaining_prob = torch.ones_like(gate_list[0].squeeze(-1))
            for idx, gate_tensor in enumerate(gate_list):
                lambda_i = torch.sigmoid(gate_tensor.squeeze(-1))
                if idx < len(gate_list) - 1:
                    p_i = lambda_i * remaining_prob
                    remaining_prob = remaining_prob * (1.0 - lambda_i)
                else:
                    p_i = remaining_prob
                pdf_list.append(p_i)
            stacked_exit_pdf = torch.stack(pdf_list, dim=2)

        expected_logits_cache: Optional[torch.Tensor] = None

        def compute_expected_logits() -> Optional[torch.Tensor]:
            nonlocal expected_logits_cache
            if expected_logits_cache is not None:
                return expected_logits_cache
            if stacked_exit_pdf is None or not hidden_states_list:
                return None
            token_exit_pdf = _select_token_positions(stacked_exit_pdf)
            expected_logits = None
            for step_idx, hidden in enumerate(hidden_states_list):
                step_hidden = _select_token_positions(hidden)
                step_logits = self.lm_head(step_hidden)
                weight = (
                    token_exit_pdf[..., step_idx].unsqueeze(-1).to(step_logits.dtype)
                )
                expected_logits = (
                    step_logits * weight
                    if expected_logits is None
                    else expected_logits + step_logits * weight
                )
            expected_logits_cache = expected_logits
            return expected_logits_cache

        logits: Optional[torch.Tensor] = None
        loss: Optional[torch.Tensor] = None

        if labels is not None:
            logits = compute_expected_logits()
            if logits is None:
                hidden_states = outputs.last_hidden_state
                logits = self.lm_head(_select_token_positions(hidden_states))
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        else:
            if stacked_exit_pdf is not None and hidden_states_list:
                if exit_at_step is not None and 0 <= exit_at_step < len(
                    hidden_states_list
                ):
                    selected_hidden = hidden_states_list[exit_at_step]
                    logits = self.lm_head(_select_token_positions(selected_hidden))
                elif exit_threshold is not None:
                    cumulative_probs = torch.cumsum(stacked_exit_pdf, dim=2)
                    threshold_value = exit_threshold
                    if isinstance(threshold_value, torch.Tensor):
                        threshold_value = threshold_value.to(cumulative_probs.device)
                    threshold_mask = cumulative_probs >= threshold_value
                    exit_steps = torch.argmax(threshold_mask.float(), dim=2)
                    last_step_idx = stacked_exit_pdf.shape[2] - 1
                    if last_step_idx >= 0:
                        never_exceeded = ~threshold_mask.any(dim=2)
                        exit_steps[never_exceeded] = last_step_idx
                    stacked_hidden = torch.stack(hidden_states_list, dim=2)
                    gather_index = (
                        exit_steps.unsqueeze(-1)
                        .unsqueeze(-1)
                        .expand(-1, -1, 1, stacked_hidden.size(-1))
                    )
                    final_hidden_states = torch.gather(
                        stacked_hidden, 2, gather_index
                    ).squeeze(2)
                    logits = self.lm_head(_select_token_positions(final_hidden_states))
                elif use_weighted_exit:
                    logits = compute_expected_logits()

            if logits is None:
                hidden_states = outputs.last_hidden_state
                logits = self.lm_head(_select_token_positions(hidden_states))

        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return result


class OuroForSequenceClassification(
    GenericForSequenceClassification, OuroPreTrainedModel
):
    pass


class OuroForTokenClassification(GenericForTokenClassification, OuroPreTrainedModel):
    pass


class OuroForQuestionAnswering(GenericForQuestionAnswering, OuroPreTrainedModel):
    base_model_prefix = (
        "transformer"  # For BC, where `transformer` was used instead of `model`
    )


__all__ = [
    "OuroPreTrainedModel",
    "OuroModel",
    "OuroForCausalLM",
    "OuroForSequenceClassification",
    "OuroForTokenClassification",
    "OuroForQuestionAnswering",
    "UniversalTransformerCache",
]
