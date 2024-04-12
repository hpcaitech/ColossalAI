try:
    import triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton is not installed. Please install Triton to use Triton kernels.")

# There may exist import error even if we have triton installed.
if HAS_TRITON:
    from .context_attn_unpad import context_attention_unpadded
    from .flash_decoding import flash_decoding_attention
    from .fused_rotary_embedding import fused_rotary_embedding
    from .kvcache_copy import copy_k_to_blocked_cache, copy_kv_to_blocked_cache
    from .no_pad_rotary_embedding import decoding_fused_rotary_embedding, rotary_embedding
    from .rms_layernorm import rms_layernorm
    from .rotary_cache_copy import get_xine_cache
    from .softmax import softmax

    __all__ = [
        "context_attention_unpadded",
        "flash_decoding_attention",
        "copy_k_to_blocked_cache",
        "copy_kv_to_blocked_cache",
        "softmax",
        "rms_layernorm",
        "rotary_embedding",
        "fused_rotary_embedding",
        "get_xine_cache",
        "decoding_fused_rotary_embedding",
    ]
