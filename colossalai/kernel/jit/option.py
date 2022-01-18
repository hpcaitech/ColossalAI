import torch

JIT_OPTIONS_SET = False


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options.
    """
    # LSG: the latest pytorch and CUDA versions may not support 
    # the following jit settings
    global JIT_OPTIONS_SET
    if JIT_OPTIONS_SET == False:
        # flags required to enable jit fusion kernels
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
            # nvfuser
            torch._C._jit_set_profiling_executor(True)
            torch._C._jit_set_profiling_mode(True)
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_override_can_fuse_on_gpu(False)
            torch._C._jit_set_texpr_fuser_enabled(False)
            torch._C._jit_set_nvfuser_enabled(True)
            torch._C._debug_set_autodiff_subgraph_inlining(False)
        else:
            # legacy pytorch fuser
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_override_can_fuse_on_cpu(True)
            torch._C._jit_override_can_fuse_on_gpu(True)

        JIT_OPTIONS_SET = True
