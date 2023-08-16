HAS_AUTO_GPTQ = False

try:
    import auto_gptq
    HAS_AUTO_GPTQ = True
except ImportError:
    warnings.warn('please install auto-gptq from https://github.com/PanQiWei/AutoGPTQ')
    HAS_AUTO_GPTQ = False

if HAS_AUTO_GPTQ: 
    from .cai_gptq import (gptq_fused_linear_triton, make_cai_quant_linear, 
                            CaiQuantLinear, CaiGPTQLinearOp)


