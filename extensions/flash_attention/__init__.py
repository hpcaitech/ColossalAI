from .flash_attention_dao_cuda import FlashAttentionDaoCudaExtension
from .flash_attention_npu import FlashAttentionNpuExtension
from .flash_attention_sdpa_cuda import FlashAttentionSdpaCudaExtension

try:
    # TODO: remove this after updating openmoe example
    import flash_attention  # noqa

    HAS_FLASH_ATTN = True
except:
    HAS_FLASH_ATTN = False


__all__ = ["FlashAttentionDaoCudaExtension", "FlashAttentionSdpaCudaExtension", "FlashAttentionNpuExtension"]
