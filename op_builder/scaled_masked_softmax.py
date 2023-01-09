import os

from .builder import Builder
from .utils import append_nvcc_threads


class ScaledMaskedSoftmaxBuilder(Builder):
    NAME = "scaled_masked_softmax"
    PREBUILT_IMPORT_PATH = "colossalai._C.scaled_masked_softmax"

    def __init__(self):
        super().__init__(name=ScaledMaskedSoftmaxBuilder.NAME, prebuilt_import_path=ScaledMaskedSoftmaxBuilder.PREBUILT_IMPORT_PATH)

    # necessary 4 functions
    def sources_files(self):
        ret = [
            self.csrc_abs_path(fname) for fname in 
            ['scaled_masked_softmax.cpp', 'scaled_masked_softmax_cuda.cu']
        ]
        return ret

    def include_dirs(self):
        return [
            self.csrc_abs_path("kernels/include"),
            self.get_cuda_home_include()
        ]

    def cxx_flags(self):
        return ['-O3'] + self.version_dependent_macros

    def nvcc_flags(self):
        extra_cuda_flags = [
            '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__', '-DTHRUST_IGNORE_CUB_VERSION_CHECK'
        ]
        ret = ['-O3', '--use_fast_math'] + self.version_dependent_macros + extra_cuda_flags
        return append_nvcc_threads(ret)
