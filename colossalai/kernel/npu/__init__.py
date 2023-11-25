try:
    import torch
    import torch_npu

    HAS_NPU = torch.npu.is_available()
except ImportError:
    HAS_NPU = False
    print("Please install torch_npu to use npu kernels.")

if HAS_NPU:
    from .mha import NPUColoAttention

    __all__ = ["HAS_NPU", "NPUColoAttention"]

else:
    __all__ = ["HAS_NPU"]
