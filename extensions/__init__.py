from .pybind.cpu_adam import CpuAdamArmExtension, CpuAdamX86Extension
from .pybind.flash_attention import (
    FlashAttentionDaoCudaExtension,
    FlashAttentionNpuExtension,
    FlashAttentionSdpaCudaExtension,
)
from .pybind.inference import InferenceOpsCudaExtension
from .pybind.layernorm import LayerNormCudaExtension
from .pybind.moe import MoeCudaExtension
from .pybind.optimizer import FusedOptimizerCudaExtension
from .pybind.softmax import ScaledMaskedSoftmaxCudaExtension, ScaledUpperTriangleMaskedSoftmaxCudaExtension

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
