import os

from .builder import Builder
from .utils import append_nvcc_threads, get_cuda_cc_flag


class MultiHeadAttnBuilder(Builder):

    NAME = "multihead_attention"
    PREBUILT_IMPORT_PATH = "colossalai._C.multihead_attention"

    def __init__(self):
        super().__init__(name=MultiHeadAttnBuilder.NAME,
        prebuilt_import_path=MultiHeadAttnBuilder.PREBUILT_IMPORT_PATH)
        

    def include_dirs(self):
        ret = [self.csrc_abs_path("kernels/include"), self.get_cuda_home_include()]
        return ret

    def sources_files(self):
        ret = [
            self.csrc_abs_path(fname) for fname in [
                'multihead_attention_1d.cpp', 'kernels/cublas_wrappers.cu', 'kernels/transform_kernels.cu',
                'kernels/dropout_kernels.cu', 'kernels/normalize_kernels.cu', 'kernels/softmax_kernels.cu',
                'kernels/general_kernels.cu', 'kernels/cuda_util.cu'
            ]
        ]
        return ret

    def cxx_flags(self):
        return ['-O3'] + self.version_dependent_macros

    def nvcc_flags(self):
        extra_cuda_flags = [
            '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__', '-DTHRUST_IGNORE_CUB_VERSION_CHECK'
        ]
        extra_cuda_flags.extend(get_cuda_cc_flag())
        ret = ['-O3', '--use_fast_math'] + self.version_dependent_macros + extra_cuda_flags
        return append_nvcc_threads(ret)
