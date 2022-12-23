import os
import sys
from pathlib import Path

from .builder import Builder


class CPUAdamBuilder(Builder):
    NAME = "cpu_adam"
    BASE_DIR = "cuda_native"

    def __init__(self):
        self.name = CPUAdamBuilder.NAME
        super().__init__()

        self.sources = [self.colossalai_src_path(path) for path in self.sources_files()]
        self.extra_include_paths = [self.colossalai_src_path(path) for path in self.include_paths()]
        self.extra_cxx_flags = ['-std=c++14', '-lcudart', '-lcublas', '-g', '-Wno-reorder', '-fopenmp', '-march=native']
        self.extra_cuda_flags = [
            '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__', '-DTHRUST_IGNORE_CUB_VERSION_CHECK'
        ]
        self.version_dependent_macros = ['-DVERSION_GE_1_1', '-DVERSION_GE_1_3', '-DVERSION_GE_1_5']

    def sources_files(self):
        return [
            os.path.join(CPUAdamBuilder.BASE_DIR, "csrc/cpu_adam.cpp"),
        ]

    def include_paths(self):
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        cuda_include = os.path.join(CUDA_HOME, "include")
        return [os.path.join(CPUAdamBuilder.BASE_DIR, "includes"), cuda_include]

    def colossalai_src_path(self, code_path):
        if os.path.isabs(code_path):
            return code_path
        else:
            return os.path.join(Path(__file__).parent.parent.absolute(), code_path)

    def strip_empty_entries(self, args):
        '''
        Drop any empty strings from the list of compile and link flags
        '''
        return [x for x in args if len(x) > 0]

    def builder(self):
        from torch.utils.cpp_extension import CUDAExtension
        return CUDAExtension(
            name=self.name,
            sources=[os.path.join('colossalai/kernel/cuda_native/csrc', path) for path in self.sources],
            include_dirs=self.extra_include_paths,
            extra_compile_args={
                'cxx': ['-O3'] + self.version_dependent_macros + self.extra_cxx_flags,
                'nvcc': ['-O3', '--use_fast_math'] + self.extra_cuda_flags
            })

    def load(self, verbose=True):
        """

        load and compile cpu_adam lib at runtime

        Args:
            verbose (bool, optional): show detailed info. Defaults to True.
        """
        import time

        from torch.utils.cpp_extension import load
        start_build = time.time()

        op_module = load(name=self.name,
                         sources=self.strip_empty_entries(self.sources),
                         extra_include_paths=self.strip_empty_entries(self.extra_include_paths),
                         extra_cflags=self.extra_cxx_flags,
                         extra_cuda_cflags=self.extra_cuda_flags,
                         extra_ldflags=[],
                         verbose=verbose)

        build_duration = time.time() - start_build
        if verbose:
            print(f"Time to load {self.name} op: {build_duration} seconds")

        return op_module
