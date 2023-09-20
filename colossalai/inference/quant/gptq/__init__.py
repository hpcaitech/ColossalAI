from .cai_gptq import HAS_AUTO_GPTQ

if HAS_AUTO_GPTQ:
    from .cai_gptq import CaiGPTQLinearOp, CaiQuantLinear
