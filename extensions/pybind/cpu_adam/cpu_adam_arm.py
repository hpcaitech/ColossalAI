import platform
from typing import List

from ...cpp_extension import _CppExtension


class CpuAdamArmExtension(_CppExtension):
    def __init__(self):
        super().__init__(name="cpu_adam_arm")

    def is_available(self) -> bool:
        # only arm allowed
        return platform.machine() == "aarch64"

    def assert_compatible(self) -> None:
        arch = platform.machine()
        assert (
            arch == "aarch64"
        ), f"[extension] The {self.name} kernel requires the CPU architecture to be aarch64 but got {arch}"

    # necessary 4 functions
    def sources_files(self):
        ret = [
            self.csrc_abs_path("kernel/arm/cpu_adam_arm.cpp"),
        ]
        return ret

    def include_dirs(self) -> List[str]:
        return super().include_dirs()

    def cxx_flags(self):
        extra_cxx_flags = [
            "-std=c++14",
            "-std=c++17",
            "-g",
            "-Wno-reorder",
            "-fopenmp",
        ]
        return ["-O3"] + self.version_dependent_macros + extra_cxx_flags

    def nvcc_flags(self):
        return []
