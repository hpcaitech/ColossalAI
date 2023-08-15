import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
from torch.utils import cpp_extension


# ninja build does not work unless include_dirs are abs path
this_dir = os.getcwd()

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def raise_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version("11.2"):
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


if not torch.cuda.is_available():
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print(
        "\nWarning: Torch did not find available GPUs on this system.\n",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
        "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
        "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None and CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version >= Version("11.8"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
        elif bare_metal_version >= Version("11.1"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
        elif bare_metal_version == Version("11.0"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"


# Check, if CUDA11 is installed for compute capability 8.0
cc_flag = []
_, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
if bare_metal_version < Version("11.0"):
    raise RuntimeError("FlashAttention is only supported on CUDA 11 and above")
# cc_flag.append("-gencode")
# cc_flag.append("arch=compute_75,code=sm_75")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")
if bare_metal_version >= Version("11.8"):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_90,code=sm_90")

llama_cuda_submodules = [
        CUDAExtension(
            name='col_fused_softmax_lib',
            sources=[
                this_dir + '/colossalai/kernel/cuda_native/csrc/attention_infer_kernels/softmax/fused_softmax.cpp', 
                this_dir + '/colossalai/kernel/cuda_native/csrc/attention_infer_kernels/softmax/scaled_masked_softmax_cuda.cu'
                ],
            extra_compile_args={
                               'cxx': ['-O3',],
                               'nvcc': append_nvcc_threads(['-O3', '--use_fast_math'] + cc_flag)
                               }
            ),
        
        CUDAExtension(
            name="col_pos_encoding_ops",
            sources=[
                this_dir + "/colossalai/kernel/cuda_native/csrc/attention_infer_kernels/rotary_embedding/pos_encoding.cpp", 
                this_dir + "/colossalai/kernel/cuda_native/csrc/attention_infer_kernels/rotary_embedding/pos_encoding_kernels.cu"
                ],
            extra_compile_args={
                               'cxx': ['-O3',],
                               'nvcc': append_nvcc_threads(['-O3', '--use_fast_math'] + cc_flag)
                                },
        ),

        CUDAExtension(
            name="col_rms_norm_ops",
            sources=[
                this_dir + "/colossalai/kernel/cuda_native/csrc/attention_infer_kernels/rmsnorm/layernorm.cpp", 
                this_dir + "/colossalai/kernel/cuda_native/csrc/attention_infer_kernels/rmsnorm/layernorm_kernels.cu"
                ],
            extra_compile_args={
                               'cxx': ['-O3',],
                               'nvcc': append_nvcc_threads(['-O3', '--use_fast_math'] + cc_flag)
                                },
            include_dirs=[
                this_dir + '/colossalai/kernel/cuda_native/csrc/attention_infer_kernels/rmsnorm',
            ],
        ),

    ]







