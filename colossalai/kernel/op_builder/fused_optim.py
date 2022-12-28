import os
import re

import torch

from .builder import Builder, get_cuda_cc_flag


class FusedOptimBuilder(Builder):
    NAME = 'fused_optim'
    BASE_DIR = "cuda_native/csrc"

    def __init__(self):
        self.name = FusedOptimBuilder.NAME
        super().__init__()

        self.extra_cxx_flags = []
        self.extra_cuda_flags = ['-lineinfo']
        self.extra_cuda_flags.extend(get_cuda_cc_flag())

        self.sources = [self.colossalai_src_path(path) for path in self.sources_files()]
        self.extra_include_paths = [self.colossalai_src_path(path) for path in self.include_paths()]
        self.version_dependent_macros = ['-DVERSION_GE_1_1', '-DVERSION_GE_1_3', '-DVERSION_GE_1_5']

    def sources_files(self):
        return [
            os.path.join(FusedOptimBuilder.BASE_DIR, fname) for fname in [
                'colossal_C_frontend.cpp', 'multi_tensor_sgd_kernel.cu', 'multi_tensor_scale_kernel.cu',
                'multi_tensor_adam.cu', 'multi_tensor_l2norm_kernel.cu', 'multi_tensor_lamb.cu'
            ]
        ]

    def include_paths(self):
        return [os.path.join(FusedOptimBuilder.BASE_DIR, "includes"), self.get_cuda_include()]

    def builder(self, name):
        from torch.utils.cpp_extension import CUDAExtension
        return CUDAExtension(
            name=name,
            sources=[os.path.join('colossalai/kernel/cuda_native/csrc', path) for path in self.sources],
            include_dirs=self.extra_include_paths,
            extra_compile_args={
                'cxx': ['-O3'] + self.version_dependent_macros + self.extra_cxx_flags,
                'nvcc': ['-O3', '--use_fast_math'] + self.extra_cuda_flags
            })
