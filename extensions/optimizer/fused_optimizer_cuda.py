from ..cuda_extension import _CudaExtension
from ..utils import get_cuda_cc_flag


class FusedOptimizerCudaExtension(_CudaExtension):
    def __init__(self):
        super().__init__(name="fused_optim_cuda")

    def sources_files(self):
        ret = [
            self.csrc_abs_path(fname)
            for fname in [
                "cuda/colossal_C_frontend.cpp",
                "cuda/multi_tensor_sgd_kernel.cu",
                "cuda/multi_tensor_scale_kernel.cu",
                "cuda/multi_tensor_adam.cu",
                "cuda/multi_tensor_l2norm_kernel.cu",
                "cuda/multi_tensor_lamb.cu",
            ]
        ]
        return ret

    def include_dirs(self):
        ret = [self.get_cuda_home_include()]
        return ret

    def cxx_flags(self):
        version_dependent_macros = ["-DVERSION_GE_1_1", "-DVERSION_GE_1_3", "-DVERSION_GE_1_5"]
        return ["-O3"] + version_dependent_macros

    def nvcc_flags(self):
        extra_cuda_flags = ["-lineinfo"]
        extra_cuda_flags.extend(get_cuda_cc_flag())
        return ["-O3", "--use_fast_math"] + extra_cuda_flags
