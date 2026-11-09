"""Temporary Ouro 5.17 diagnostic adapters, scoped to one imported module.

Not production support: only default full-head RoPE and no-cache eager prefill
with contiguous positions are covered. Original module bindings are restored.
"""

from contextlib import contextmanager

import torch
import transformers


@contextmanager
def ouro_compat(module, config, mode):
    if transformers.__version__ != "5.17.0":
        raise ValueError("This diagnostic adapter is pinned to Transformers 5.17.0")
    assert mode in {"rope-registry", "rope", "rope-mask"}
    assert config.rope_parameters["rope_type"] == "default"
    assert config.rope_parameters.get("partial_rotary_factor", 1.0) == 1.0
    assert not config.use_cache and not config.use_sliding_window
    assert config._attn_implementation == "eager"

    # Reuse HF's standard full-head RoPE computation, without changing its registry.
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    original_registry = module.ROPE_INIT_FUNCTIONS
    original_mask = module.create_causal_mask
    rotary_class = module.OuroRotaryEmbedding
    missing = object()
    original_default = rotary_class.__dict__.get("compute_default_rope_parameters", missing)
    assert "default" not in original_registry

    def prefill_mask(*, input_embeds, cache_position, **kwargs):
        if kwargs.get("past_key_values") is not None:
            raise ValueError("Diagnostic mask adapter does not support KV cache")
        expected = torch.arange(input_embeds.shape[1], device=input_embeds.device)
        if not torch.equal(cache_position, expected):
            raise ValueError("Diagnostic adapter requires contiguous prefill positions")
        if not torch.equal(kwargs["position_ids"], expected.unsqueeze(0)):
            raise ValueError("Diagnostic adapter requires the default position_ids")
        return original_mask(inputs_embeds=input_embeds, **kwargs)

    try:
        module.ROPE_INIT_FUNCTIONS = {
            **original_registry,
            "default": Qwen2RotaryEmbedding.compute_default_rope_parameters,
        }
        if mode != "rope-registry":
            rotary_class.compute_default_rope_parameters = staticmethod(
                Qwen2RotaryEmbedding.compute_default_rope_parameters
            )
        if mode == "rope-mask":
            module.create_causal_mask = prefill_mask
        yield
    finally:
        module.ROPE_INIT_FUNCTIONS = original_registry
        module.create_causal_mask = original_mask
        if original_default is missing:
            if "compute_default_rope_parameters" in rotary_class.__dict__:
                delattr(rotary_class, "compute_default_rope_parameters")
        else:
            rotary_class.compute_default_rope_parameters = original_default
