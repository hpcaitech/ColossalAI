import os

from .builder import Builder, get_cuda_cc_flag


class MultiHeadAttnBuilder(Builder):

    def __init__(self):
        self.base_dir = "cuda_native/csrc"
        self.name = 'multihead_attention'
        super().__init__()
        self.extra_cxx_flags = []
        self.extra_cuda_flags = [
            '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__', '-DTHRUST_IGNORE_CUB_VERSION_CHECK'
        ]

        self.extra_cuda_flags.extend(get_cuda_cc_flag())
        self.sources = [self.colossalai_src_path(path) for path in self.sources_files()]
        self.extra_include_paths = [self.colossalai_src_path(path) for path in self.include_paths()]

        self.version_dependent_macros = ['-DVERSION_GE_1_1', '-DVERSION_GE_1_3', '-DVERSION_GE_1_5']

    def sources_files(self):
        return [
            os.path.join(self.base_dir, fname) for fname in [
                'multihead_attention_1d.cpp', 'kernels/cublas_wrappers.cu', 'kernels/transform_kernels.cu',
                'kernels/dropout_kernels.cu', 'kernels/normalize_kernels.cu', 'kernels/softmax_kernels.cu',
                'kernels/general_kernels.cu', 'kernels/cuda_util.cu'
            ]
        ]

    def include_paths(self):
        from torch.utils.cpp_extension import CUDA_HOME
        ret = []
        cuda_include = os.path.join(CUDA_HOME, "include")
        ret = [os.path.join(self.base_dir, "includes"), cuda_include]
        ret.append(os.path.join(self.base_dir, "kernels", "include"))
        print("include_paths", ret)
        return ret

    def builder(self, name):
        from torch.utils.cpp_extension import CUDAExtension
        return CUDAExtension(
            name=name,
            sources=[os.path.join('colossalai/kernel/cuda_native/csrc', path) for path in self.sources],
            include_dirs=self.extra_include_paths,
            extra_compile_args={
                'cxx': ['-O3'] + self.version_dependent_macros,
                'nvcc': ['-O3', '--use_fast_math'] + self.extra_cuda_flags
            })
