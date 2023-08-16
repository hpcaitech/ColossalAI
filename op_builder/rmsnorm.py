import os

from .builder import Builder
from .utils import append_nvcc_threads, get_cuda_cc_flag

class RMSNORMBuilder(Builder):

    NAME = "rmsnorm"
    PREBUILT_IMPORT_PATH = "colossalai._C.rmsnorm"

    def __init__(self):
        super().__init__(name=RMSNORMBuilder.NAME,
        prebuilt_import_path=RMSNORMBuilder.PREBUILT_IMPORT_PATH)
        

    def include_dirs(self):
        ret = [self.csrc_abs_path("attention_infer_kernels/rmsnorm"), self.get_cuda_home_include()]
        return ret

    def sources_files(self):
        ret = [
            self.csrc_abs_path(fname) for fname in [
                'attention_infer_kernels/rmsnorm/layernorm_kernels.cu', 
                'attention_infer_kernels/rmsnorm/layernorm.cpp'
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
