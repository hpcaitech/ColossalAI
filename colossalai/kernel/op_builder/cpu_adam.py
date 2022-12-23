import os
import sys

from .builder import CUDAOpBuilder


class CPUAdamBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"
    BASE_DIR = "cuda_native"

    def __init__(self):
        super().__init__(name=self.NAME)

    def is_compatible(self):
        # Disable on Windows.
        return sys.platform != "win32"

    def absolute_name(self):
        return f"patrickstar.ops.adam.{self.NAME}_op"

    def sources(self):
        return [
            os.path.join(CPUAdamBuilder.BASE_DIR, "csrc/cpu_adam.cpp"),
        ]

    def include_paths(self):
        import torch

        cuda_include = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")
        return [os.path.join(CPUAdamBuilder.BASE_DIR, "includes"), cuda_include]

    def cxx_args(self):
        import torch

        cuda_lib64 = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "lib64")
        cpu_arch = self.cpu_arch()
        simd_width = self.simd_width()

        return [
            "-O3",
            "-std=c++14",
            f"-L{cuda_lib64}",
            "-lcudart",
            "-lcublas",
            "-g",
            "-Wno-reorder",
            cpu_arch,
            "-fopenmp",
            simd_width,
        ]
