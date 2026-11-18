"""Ouro's remote-code training path on Transformers 5.17.

The official decoder, universal-transformer loop, exit gate and LM loss are
inherited unchanged. Only construction and the prefill mask boundary differ.
No Hugging Face module, registry or cached source is patched.

Construction below follows ByteDance's Apache-2.0 Ouro implementation:
https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py
"""

import sys
from functools import lru_cache

import torch
from torch import nn

__all__ = ["get_ouro_model_class"]


def _check_config(config):
    from transformers import __version__

    if __version__ != "5.17.0":
        raise NotImplementedError(f"Ouro compatibility is validated on transformers==5.17.0, installed {__version__}")
    rope = getattr(config, "rope_parameters", None) or {}
    if rope.get("rope_type", "default") != "default" or rope.get("partial_rotary_factor", 1.0) != 1.0:
        raise NotImplementedError("Ouro compatibility supports only default, full-head RoPE")
    if config.use_cache or config.use_sliding_window or any(t != "full_attention" for t in config.layer_types):
        raise NotImplementedError(
            "Ouro compatibility requires use_cache=False and full attention without sliding windows"
        )
    if config._attn_implementation != "eager":
        raise NotImplementedError("Ouro compatibility requires attn_implementation='eager'")
    if config.head_dim != config.hidden_size // config.num_attention_heads or config.head_dim % 2:
        raise NotImplementedError("Ouro compatibility requires an even, full attention head dimension")


def _default_rope_parameters(config, device=None, seq_len=None):
    # Standard RoPE: frequency[i] = theta ** (-2*i/head_dim).
    # HF5 calls this method again during post_init / from_pretrained, including
    # when construction took place on the meta device.
    theta = config.rope_parameters["rope_theta"]
    dim = config.head_dim
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return inv_freq, 1.0


@lru_cache(maxsize=None)
def get_ouro_model_class(original_class):
    """Return an isolated subclass of an already trusted official Ouro LM class.

    Load the official class from a pinned revision with HF's
    get_class_from_dynamic_module, then call this function before construction.
    The returned class supports both Class(config) and
    Class.from_pretrained(path, config=config, attn_implementation="eager").
    Set config.use_cache=False. The validated scope is no-cache eager prefill
    with contiguous positions and default full-head RoPE; see
    docs/HF5-COMPATIBILITY.md.

    Save weights with save_pretrained and reload through this factory again.
    AutoModel alone does not select the adapter. Whole-model Python pickling is
    not supported; use state_dict / save_pretrained instead.
    """
    module = sys.modules[original_class.__module__]
    if original_class is not getattr(module, "OuroForCausalLM", None):
        raise TypeError("Pass the official remote-code OuroForCausalLM class")

    class OuroRotaryEmbedding(module.OuroRotaryEmbedding):
        compute_default_rope_parameters = staticmethod(_default_rope_parameters)

        def __init__(self, config, device=None):
            nn.Module.__init__(self)
            self.config = config
            self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings
            self.rope_init_fn = self.compute_default_rope_parameters
            inv_freq, self.attention_scaling = self.rope_init_fn(config, device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.original_inv_freq = self.inv_freq

    class OuroModel(module.OuroModel):
        def __init__(self, config):
            _check_config(config)
            module.OuroPreTrainedModel.__init__(self, config)
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.layers = nn.ModuleList(
                [module.OuroDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
            self.norm = module.OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.rotary_emb = OuroRotaryEmbedding(config)
            self.gradient_checkpointing = False
            self.has_sliding_layers = False
            self.total_ut_steps = getattr(config, "total_ut_steps", 4)
            self.early_exit_gate = nn.Linear(config.hidden_size, 1)
            self.post_init()

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            cache_position=None,
            **kwargs,
        ):
            from transformers.masking_utils import create_causal_mask

            _check_config(self.config)
            if use_cache or past_key_values is not None:
                raise NotImplementedError("Ouro compatibility does not support KV cache")
            if self.gradient_checkpointing:
                raise NotImplementedError("Ouro gradient checkpointing has not been validated")
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("Specify exactly one of input_ids or inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            batch, length = inputs_embeds.shape[:2]
            expected = torch.arange(length, device=inputs_embeds.device)
            if cache_position is not None and not torch.equal(cache_position, expected):
                raise NotImplementedError("Ouro compatibility requires contiguous cache_position starting at zero")
            if position_ids is None:
                position_ids = expected.unsqueeze(0)
            if position_ids.shape not in {(1, length), (batch, length)} or not torch.equal(
                position_ids, expected.unsqueeze(0).expand_as(position_ids)
            ):
                raise NotImplementedError("Ouro compatibility requires contiguous position_ids starting at zero")
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != (batch, length):
                    raise NotImplementedError("Ouro compatibility accepts only a 2D padding attention_mask")
                if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                    raise ValueError("The 2D attention_mask must contain only zeros and ones")
            mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            # The official forward already accepts a mapping of prepared masks.
            # This bypasses only its old mask API call, retaining the full loop.
            return super().forward(
                inputs_embeds=inputs_embeds,
                attention_mask={"full_attention": mask},
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=expected,
                **kwargs,
            )

    class OuroForCausalLM(original_class):
        # HF5 represents tied-weight relationships as a mapping, not a list.
        _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
        # The adapter is provided by ColossalAI, not a standalone HF remote-code file.
        _auto_class = None

        def __init__(self, config):
            _check_config(config)
            module.OuroPreTrainedModel.__init__(self, config)
            self.model = OuroModel(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.chunk_size = getattr(config, "chunk_size", 2)
            self.early_exit_step = getattr(config, "early_exit_step", None)
            self.early_exit_threshold = getattr(config, "early_exit_threshold", None)
            self.post_init()

    # Stable names let ShardFormer select a policy without registering each HF
    # snapshot hash. Classes remain cached separately for each original class.
    OuroForCausalLM.__qualname__ = "OuroForCausalLM"
    OuroModel.__qualname__ = "OuroModel"
    return OuroForCausalLM
