"""
Utils for model inference
"""

import os

import torch

from colossalai.kernel.triton.copy_kv_cache_dest import copy_kv_cache_to_dest


def copy_kv_to_mem_cache(layer_id, key_buffer, value_buffer, context_mem_index, mem_manager):
    """
    This function copies the key and value cache to the memory cache
    Args:
        layer_id : id of current layer
        key_buffer : key cache
        value_buffer : value cache
        context_mem_index : index of memory cache in kv cache manager
        mem_manager : cache manager
    """
    copy_kv_cache_to_dest(key_buffer, context_mem_index, mem_manager.key_buffer[layer_id])
    copy_kv_cache_to_dest(value_buffer, context_mem_index, mem_manager.value_buffer[layer_id])


def init_to_get_rotary(self, base=10000, use_elem=False):
    """
    This function initializes the rotary positional embedding, it is compatible for all models and is called in ShardFormer
    Args:
        self : Model that holds the rotary positional embedding
        base : calculation arg
        use_elem : activated when using chatglm-based models
    """
    self.config.head_dim_ = self.config.hidden_size // self.config.num_attention_heads
    if not hasattr(self.config, "rope_scaling"):
        rope_scaling_factor = 1.0
    else:
        rope_scaling_factor = self.config.rope_scaling.factor if self.config.rope_scaling is not None else 1.0

    if hasattr(self.config, "max_sequence_length"):
        max_seq_len = self.config.max_sequence_length
    elif hasattr(self.config, "max_position_embeddings"):
        max_seq_len = self.config.max_position_embeddings * rope_scaling_factor
    else:
        max_seq_len = 2048 * rope_scaling_factor
    base = float(base)

    # NTK  ref: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    ntk_alpha = os.environ.get("INFER_NTK_ALPHA", None)

    if ntk_alpha is not None:
        ntk_alpha = float(ntk_alpha)
        assert ntk_alpha >= 1, "NTK alpha must be greater than or equal to 1"
        if ntk_alpha > 1:
            print(f"Note: NTK enabled, alpha set to {ntk_alpha}")
        max_seq_len *= ntk_alpha
        base = base * (ntk_alpha ** (self.head_dim_ / (self.head_dim_ - 2)))  # Base change formula

    n_elem = self.config.head_dim_
    if use_elem:
        n_elem //= 2

    inv_freq = 1.0 / (base ** (torch.arange(0, n_elem, 2, device="cpu", dtype=torch.float32) / n_elem))
    t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
    freqs = torch.outer(t, inv_freq)

    self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
    self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
