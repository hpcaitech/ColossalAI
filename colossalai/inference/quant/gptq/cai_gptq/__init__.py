import warnings

HAS_AUTO_GPTQ = False
try:
    import auto_gptq

    HAS_AUTO_GPTQ = True
except ImportError:
    warnings.warn("please install auto-gptq from https://github.com/PanQiWei/AutoGPTQ")
    HAS_AUTO_GPTQ = False

if HAS_AUTO_GPTQ:
    from .cai_quant_linear import CaiQuantLinear, ColCaiQuantLinear, RowCaiQuantLinear
    from .gptq_op import CaiGPTQLinearOp
