from .flash_attention_cuda import FlashAttentionCudaExtension
from .flash_attention_npu import FlashAttentionNpuExtension

try:
    import flash_attention

    HAS_FLASH_ATTN = True
except:
    HAS_FLASH_ATTN = False

# we do not use xformers anymore
HAS_MEM_EFF_ATTN = False

__all__ = ["FlashAttentionCudaExtension", "FlashAttentionNpuExtension"]
