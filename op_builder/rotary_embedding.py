import os
from packaging.version import parse, Version
from setuptools import setup, find_packages
import subprocess

from .builder import Builder
from .utils import append_nvcc_threads, get_cuda_cc_flag

from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

def add_cc_flags():
    def get_cuda_bare_metal_version(cuda_dir):
        raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
        output = raw_output.split()
        release_idx = output.index("release") + 1
        bare_metal_version = parse(output[release_idx].split(",")[0])

        return raw_output, bare_metal_version

    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    print(bare_metal_version)
    if bare_metal_version < Version("11.0"):
        raise RuntimeError("FlashAttention is only supported on CUDA 11 and above")

    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")
    
    return cc_flag

class ROTARYEMBEDDINGBuilder(Builder):

    NAME = "rotary_embedding"
    PREBUILT_IMPORT_PATH = "colossalai._C.rotary_embedding"

    def __init__(self):
        super().__init__(name=ROTARYEMBEDDINGBuilder.NAME,
        prebuilt_import_path=ROTARYEMBEDDINGBuilder.PREBUILT_IMPORT_PATH)
        

    def include_dirs(self):
        ret = [self.csrc_abs_path("attention_infer_kernels/rotary_embedding"), self.get_cuda_home_include()]
        return ret

    def sources_files(self):
        ret = [
            self.csrc_abs_path(fname) for fname in [
                'attention_infer_kernels/rotary_embedding/pos_encoding_kernels.cu', 
                'attention_infer_kernels/rotary_embedding/pos_encoding.cpp'
            ]
        ]
        return ret

    def cxx_flags(self):
        return ['-O3'] + self.version_dependent_macros

    def nvcc_flags(self):
        extra_cuda_flags = [
            '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__', '-DTHRUST_IGNORE_CUB_VERSION_CHECK'
        ]
        ret = ['-O3', '--use_fast_math'] + self.version_dependent_macros + extra_cuda_flags
        return append_nvcc_threads(ret) + add_cc_flags()
