import os

from .builder import Builder, get_cuda_cc_flag


class MultiHeadAttnBuilder(Builder):

    def __init__(self):
        self.base_dir = "cuda_native/csrc"
        self.name = 'multihead_attention'
        super().__init__()

        self.version_dependent_macros = ['-DVERSION_GE_1_1', '-DVERSION_GE_1_3', '-DVERSION_GE_1_5']

    def include_dirs(self):
        ret = []
        ret = [os.path.join(self.base_dir, "includes"), self.get_cuda_home_include()]
        ret.append(os.path.join(self.base_dir, "kernels", "include"))
        return [self.colossalai_src_path(path) for path in ret]

    def sources_files(self):
        ret = [
            os.path.join(self.base_dir, fname) for fname in [
                'multihead_attention_1d.cpp', 'kernels/cublas_wrappers.cu', 'kernels/transform_kernels.cu',
                'kernels/dropout_kernels.cu', 'kernels/normalize_kernels.cu', 'kernels/softmax_kernels.cu',
                'kernels/general_kernels.cu', 'kernels/cuda_util.cu'
            ]
        ]
        return [self.colossalai_src_path(path) for path in ret]

    def cxx_flags(self):
        return ['-O3'] + self.version_dependent_macros

    def nvcc_flags(self):
        extra_cuda_flags = [
            '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__', '-DTHRUST_IGNORE_CUB_VERSION_CHECK'
        ]
        extra_cuda_flags.extend(get_cuda_cc_flag())
        ret = ['-O3', '--use_fast_math'] + extra_cuda_flags
        return ret
