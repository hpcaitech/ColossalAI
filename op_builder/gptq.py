import re

import torch

from .builder import Builder
from .utils import append_nvcc_threads


class GPTQBuilder(Builder):
    NAME = "cu_gptq"
    PREBUILT_IMPORT_PATH = "colossalai._C.cu_gptq"

    def __init__(self):
        super().__init__(name=GPTQBuilder.NAME, prebuilt_import_path=GPTQBuilder.PREBUILT_IMPORT_PATH)

    def include_dirs(self):
        ret = [self.csrc_abs_path("gptq"), self.get_cuda_home_include()]
        return ret

    def sources_files(self):
        ret = [
            self.csrc_abs_path(fname)
            for fname in [
                "gptq/linear_gptq.cpp",
                "gptq/column_remap.cu",
                "gptq/cuda_buffers.cu",
                "gptq/q4_matmul.cu",
                "gptq/q4_matrix.cu",
            ]
        ]
        return ret

    def cxx_flags(self):
        return ["-O3"] + self.version_dependent_macros

    def nvcc_flags(self):
        extra_cuda_flags = [
            "-v",
            "-std=c++14",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
            "-lcublas",
        ]

        for arch in torch.cuda.get_arch_list():
            res = re.search(r"sm_(\d+)", arch)
            if res:
                arch_cap = res[1]
                if int(arch_cap) >= 80:
                    extra_cuda_flags.extend(["-gencode", f"arch=compute_{arch_cap},code={arch}"])

        ret = ["-O3", "--use_fast_math"] + self.version_dependent_macros + extra_cuda_flags
        return append_nvcc_threads(ret)
