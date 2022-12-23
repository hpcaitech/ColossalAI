from .cuda_native import FusedScaleMaskSoftmax, LayerNorm, MultiHeadAttention

try:
    from colossalai._C import fused_optim
except:
    from colossalai.kernel.op_builder.fused_optim import FusedOptimBuilder
    fused_optim = FusedOptimBuilder().load()

try:
    from colossalai._C import cpu_optim
except ImportError:
    from colossalai.kernel.op_builder import CPUAdamBuilder
    cpu_optim = CPUAdamBuilder().load()

__all__ = ["fused_optim", "cpu_optim", "LayerNorm", "FusedScaleMaskSoftmax", "MultiHeadAttention"]
