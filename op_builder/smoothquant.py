import torch

from .builder import Builder
from .utils import append_nvcc_threads


class SmoothquantBuilder(Builder):
    NAME = "cu_smoothquant"
    PREBUILT_IMPORT_PATH = "colossalai._C.cu_smoothquant"

    def __init__(self):
        super().__init__(name=SmoothquantBuilder.NAME, prebuilt_import_path=SmoothquantBuilder.PREBUILT_IMPORT_PATH)

    def include_dirs(self):
        ret = [self.csrc_abs_path("smoothquant"), self.get_cuda_home_include()]
        return ret

    def sources_files(self):
        ret = [
            self.csrc_abs_path(fname)
            for fname in [
                "smoothquant/binding.cpp",
                "smoothquant/linear.cu",
            ]
        ]
        return ret

    def cxx_flags(self):
        return ["-O3"] + self.version_dependent_macros

    def nvcc_flags(self):
        compute_capability = torch.cuda.get_device_capability()
        cuda_arch = compute_capability[0] * 100 + compute_capability[1] * 10

        extra_cuda_flags = [
            "-v",
            f"-DCUDA_ARCH={cuda_arch}",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
        ]

        ret = ["-O3", "--use_fast_math"] + self.version_dependent_macros + extra_cuda_flags
        return append_nvcc_threads(ret)

    def builder(self):
        try:
            super().builder()
        except:
            warnings.warn("build smoothquant lib not successful")
