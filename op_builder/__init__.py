from .cpu_adam import CPUAdamBuilder
from .fused_optim import FusedOptimBuilder
from .layernorm import LayerNormBuilder
from .moe import MOEBuilder
from .multi_head_attn import MultiHeadAttnBuilder
from .scaled_masked_softmax import ScaledMaskedSoftmaxBuilder
from .scaled_upper_triangle_masked_softmax import ScaledUpperTrainglemaskedSoftmaxBuilder
from .rmsnorm import RMSNORMBuilder
from .rotary_embedding import ROTARYEMBEDDINGBuilder

ALL_OPS = {
    'cpu_adam': CPUAdamBuilder,
    'fused_optim': FusedOptimBuilder,
    'moe': MOEBuilder,
    'multi_head_attn': MultiHeadAttnBuilder,
    'scaled_masked_softmax': ScaledMaskedSoftmaxBuilder,
    'scaled_upper_triangle_masked_softmax': ScaledUpperTrainglemaskedSoftmaxBuilder,
    'layernorm': LayerNormBuilder,
    'rmsnorm': RMSNORMBuilder,
    'rotary_embedding': ROTARYEMBEDDINGBuilder,
}

__all__ = [
    'ALL_OPS', 'CPUAdamBuilder', 'FusedOptimBuilder', 'MultiHeadAttnBuilder', 'ScaledMaskedSoftmaxBuilder',
    'ScaledUpperTrainglemaskedSoftmaxBuilder', 'MOEBuilder', 'MultiTensorSGDBuilder', 'MultiTensorAdamBuilder',
    'MultiTensorLambBuilder', 'MultiTensorScaleBuilder', 'MultiTensorL2NormBuilder', 
    'RMSNORMBuilder', 
    'ROTARYEMBEDDINGBuilder',
]
