import os

from .builder import Builder
from .utils import append_nvcc_threads, get_cuda_cc_flag


class MOEBuilder(Builder):

    NAME = "moe"
    PREBUILT_IMPORT_PATH = "colossalai._C.moe"

    def __init__(self):
        super().__init__(name=MOEBuilder.NAME, prebuilt_import_path=MOEBuilder.PREBUILT_IMPORT_PATH)

    def include_dirs(self):
        ret = [
            self.csrc_abs_path("kernels/include"),
            self.get_cuda_home_include()
        ]
        return ret

    def sources_files(self):
        ret = [self.csrc_abs_path(fname) for fname in ['moe_cuda.cpp', 'moe_cuda_kernel.cu']]
        return ret

    def cxx_flags(self):
        return ['-O3'] + self.version_dependent_macros

    def nvcc_flags(self):
        extra_cuda_flags = [
            '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '--expt-relaxed-constexpr',
            '--expt-extended-lambda'
        ]
        extra_cuda_flags.extend(get_cuda_cc_flag())
        ret = ['-O3', '--use_fast_math'] + extra_cuda_flags
        return append_nvcc_threads(ret)
