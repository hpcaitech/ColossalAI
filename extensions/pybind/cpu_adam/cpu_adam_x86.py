import platform

from ...cuda_extension import _CudaExtension
from ...utils import append_nvcc_threads


class CpuAdamX86Extension(_CudaExtension):
    def __init__(self):
        super().__init__(name="cpu_adam_x86")

    def is_available(self) -> bool:
        return platform.machine() == "x86_64" and super().is_available()

    def assert_compatible(self) -> None:
        arch = platform.machine()
        assert (
            arch == "x86_64"
        ), f"[extension] The {self.name} kernel requires the CPU architecture to be x86_64 but got {arch}"
        super().assert_compatible()

    # necessary 4 functions
    def sources_files(self):
        ret = [
            self.csrc_abs_path("kernel/x86/cpu_adam.cpp"),
        ]
        return ret

    def cxx_flags(self):
        extra_cxx_flags = [
            "-std=c++14",
            "-std=c++17",
            "-lcudart",
            "-lcublas",
            "-g",
            "-Wno-reorder",
            "-fopenmp",
            "-march=native",
        ]
        return ["-O3"] + self.version_dependent_macros + extra_cxx_flags

    def nvcc_flags(self):
        extra_cuda_flags = [
            "-std=c++14",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
        ]
        ret = ["-O3", "--use_fast_math"] + self.version_dependent_macros + extra_cuda_flags + super().nvcc_flags()
        return append_nvcc_threads(ret)
