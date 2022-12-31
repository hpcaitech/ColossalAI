import os

from .builder import Builder, get_cuda_cc_flag


class FusedOptimBuilder(Builder):
    NAME = 'fused_optim'
    BASE_DIR = "cuda_native/csrc"

    def __init__(self):
        self.name = FusedOptimBuilder.NAME
        super().__init__()
        self.version_dependent_macros = ['-DVERSION_GE_1_1', '-DVERSION_GE_1_3', '-DVERSION_GE_1_5']

    def sources_files(self):
        ret = [
            self.colossalai_src_path(os.path.join(FusedOptimBuilder.BASE_DIR, fname)) for fname in [
                'colossal_C_frontend.cpp', 'multi_tensor_sgd_kernel.cu', 'multi_tensor_scale_kernel.cu',
                'multi_tensor_adam.cu', 'multi_tensor_l2norm_kernel.cu', 'multi_tensor_lamb.cu'
            ]
        ]
        return ret

    def include_dirs(self):
        ret = [os.path.join(FusedOptimBuilder.BASE_DIR, "includes"), self.get_cuda_home_include()]
        return [self.colossalai_src_path(path) for path in ret]

    def cxx_flags(self):
        extra_cxx_flags = []
        return ['-O3'] + self.version_dependent_macros + extra_cxx_flags

    def nvcc_flags(self):
        extra_cuda_flags = ['-lineinfo']
        extra_cuda_flags.extend(get_cuda_cc_flag())
        return ['-O3', '--use_fast_math'] + extra_cuda_flags
