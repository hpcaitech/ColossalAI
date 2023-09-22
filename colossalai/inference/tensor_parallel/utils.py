import types
import warnings

import torch
try:
    from vllm.model_executor.models.llama import LlamaAttention
    VLLM_INSTALLED = True
except ImportError:
    warnings.warn("vllm is not installed, PageAttention will not be replaced.")
    VLLM_INSTALLED = False

def init_to_get_rotary(self, base=10000):
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
    inv_freq = 1.0 / (base**(torch.arange(0, self.config.head_dim_, 2, device="cpu", dtype=torch.float32) /
                             self.config.head_dim_))
    t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
    freqs = torch.outer(t, inv_freq)

    self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
    self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
    return


def replace_page_attention(model, kv_cache_stream):
    if VLLM_INSTALLED:
        
        from colossalai.inference.continous_batching.layers.attention import PagedAttentionWithRoPE
        
        layers = model.model.layers
        for i in range(len(layers)):
            layer = layers[i]
            if isinstance(layer.self_attn, LlamaAttention) is True:
                attn = PagedAttentionWithRoPE(layer.self_attn.num_heads,
                                            layer.self_attn.head_dim,
                                            layer.self_attn.scaling,
                                            rotary_dim=layer.self_attn.head_dim,
                                            num_kv_heads=layer.self_attn.num_kv_heads,
                                            kv_cache_stream=kv_cache_stream)
                setattr(layer.self_attn, 'attn', attn)

    return model
