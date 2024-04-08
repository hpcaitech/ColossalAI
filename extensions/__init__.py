from .cpu_adam import CpuAdamArmExtension, CpuAdamX86Extension
from .flash_attention import FlashAttentionDaoCudaExtension, FlashAttentionNpuExtension, FlashAttentionSdpaCudaExtension
from .inference import InferenceOpsCudaExtension
from .layernorm import LayerNormCudaExtension
from .moe import MoeCudaExtension
from .optimizer import FusedOptimizerCudaExtension
from .softmax import ScaledMaskedSoftmaxCudaExtension, ScaledUpperTriangleMaskedSoftmaxCudaExtension

ALL_EXTENSIONS = [
    CpuAdamArmExtension,
    CpuAdamX86Extension,
    LayerNormCudaExtension,
    MoeCudaExtension,
    FusedOptimizerCudaExtension,
    InferenceOpsCudaExtension,
    ScaledMaskedSoftmaxCudaExtension,
    ScaledUpperTriangleMaskedSoftmaxCudaExtension,
    FlashAttentionDaoCudaExtension,
    FlashAttentionSdpaCudaExtension,
    FlashAttentionNpuExtension,
]

__all__ = [
    "CpuAdamArmExtension",
    "CpuAdamX86Extension",
    "LayerNormCudaExtension",
    "MoeCudaExtension",
    "FusedOptimizerCudaExtension",
    "InferenceOpsCudaExtension",
    "ScaledMaskedSoftmaxCudaExtension",
    "ScaledUpperTriangleMaskedSoftmaxCudaExtension",
    "FlashAttentionDaoCudaExtension",
    "FlashAttentionSdpaCudaExtension",
    "FlashAttentionNpuExtension",
]
