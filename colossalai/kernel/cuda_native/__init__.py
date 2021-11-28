from .builder import _build_cuda_native_kernel

CUDA_NATIVE_KERNEL_BUILD = False


def build_cuda_native_kernel():
    global CUDA_NATIVE_KERNEL_BUILD
    if CUDA_NATIVE_KERNEL_BUILD == False:
        _build_cuda_native_kernel()
        CUDA_NATIVE_KERNEL_BUILD = True


build_cuda_native_kernel()

from .layer_norm import MixedFusedLayerNorm as LayerNorm
from .scaled_softmax import FusedScaleMaskSoftmax
from .multihead_attention import MultiHeadAttention