from ..cuda_extension import _CudaExtension
from ..utils import append_nvcc_threads, get_cuda_cc_flag


class LayerNormCudaExtension(_CudaExtension):
    def __init__(self):
        super().__init__(name="layernorm_cuda")

    def sources_files(self):
        ret = [self.csrc_abs_path(fname) for fname in ["cuda/layer_norm_cuda.cpp", "cuda/layer_norm_cuda_kernel.cu"]]
        return ret

    def include_dirs(self):
        ret = [self.get_cuda_home_include()]
        return ret

    def cxx_flags(self):
        return ["-O3"] + self.version_dependent_macros

    def nvcc_flags(self):
        extra_cuda_flags = ["-maxrregcount=50"]
        extra_cuda_flags.extend(get_cuda_cc_flag())
        ret = ["-O3", "--use_fast_math"] + extra_cuda_flags + self.version_dependent_macros
        return append_nvcc_threads(ret)
