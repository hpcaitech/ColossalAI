import os

from .builder import Builder
from .utils import append_nvcc_threads


class CPUAdamBuilder(Builder):
    NAME = "cpu_adam"
    BASE_DIR = "cuda_native"

    def __init__(self):
        self.name = CPUAdamBuilder.NAME
        super().__init__()

        self.version_dependent_macros = ['-DVERSION_GE_1_1', '-DVERSION_GE_1_3', '-DVERSION_GE_1_5']

    # necessary 4 functions
    def sources_files(self):
        ret = [
            os.path.join(CPUAdamBuilder.BASE_DIR, "csrc/cpu_adam.cpp"),
        ]
        return [self.colossalai_src_path(path) for path in ret]

    def include_dirs(self):
        return [
            self.colossalai_src_path(os.path.join(CPUAdamBuilder.BASE_DIR, "includes")),
            self.get_cuda_home_include()
        ]

    def cxx_flags(self):
        extra_cxx_flags = ['-std=c++14', '-lcudart', '-lcublas', '-g', '-Wno-reorder', '-fopenmp', '-march=native']
        return ['-O3'] + self.version_dependent_macros + extra_cxx_flags

    def nvcc_flags(self):
        extra_cuda_flags = [
            '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__', '-DTHRUST_IGNORE_CUB_VERSION_CHECK'
        ]

        return append_nvcc_threads(['-O3', '--use_fast_math'] + self.version_dependent_macros + extra_cuda_flags)

    # necessary 4 functions
