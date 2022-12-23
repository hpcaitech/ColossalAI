import os

from .builder import Builder, get_cuda_cc_flag


class MOEBuilder(Builder):

    def __init__(self):
        self.base_dir = "cuda_native/csrc"
        self.name = 'moe'
        super().__init__()

    def include_dirs(self):
        ret = []
        ret = [os.path.join(self.base_dir, "includes"), self.get_cuda_home_include()]
        ret.append(os.path.join(self.base_dir, "kernels", "include"))
        return [self.colossalai_src_path(path) for path in ret]

    def sources_files(self):
        ret = [os.path.join(self.base_dir, fname) for fname in ['moe_cuda.cpp', 'moe_cuda_kernel.cu']]
        return [self.colossalai_src_path(path) for path in ret]

    def cxx_flags(self):
        return ['-O3', '-DVERSION_GE_1_1', '-DVERSION_GE_1_3', '-DVERSION_GE_1_5']

    def nvcc_flags(self):
        extra_cuda_flags = [
            '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '--expt-relaxed-constexpr',
            '--expt-extended-lambda'
        ]
        extra_cuda_flags.extend(get_cuda_cc_flag())
        ret = ['-O3', '--use_fast_math'] + extra_cuda_flags
        return ret
