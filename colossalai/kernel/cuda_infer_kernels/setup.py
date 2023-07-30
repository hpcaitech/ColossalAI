# Adapted from https://github.com/NVIDIA/apex/blob/master/setup.py
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

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_version != torch_binary_version):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )


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


print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

cmdclass = {}
ext_modules = []

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

raise_if_cuda_home_none("flash_attn")
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

setup(
    name='colossal-cuda-kernels',
    ext_modules=[
        CUDAExtension(
            name='col_fused_softmax_lib',
            sources=['softmax/fused_softmax.cpp', 'softmax/scaled_masked_softmax_cuda.cu'],
            extra_compile_args={
                               'cxx': ['-O3',],
                               'nvcc': append_nvcc_threads(['-O3', '--use_fast_math'] + cc_flag)
                               }
            ),
        
        CUDAExtension(
                name="col_flash_attn_2_lib",
                sources=[
                    "flash_attn/flash_api.cpp",
                    "flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim224_fp16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim224_bf16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu",
                    "flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu",
                ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        "--ptxas-options=-v",
                        "-lineinfo"
                    ]
                    + generator_flag
                    + cc_flag
                ),
            },
            include_dirs=[
                Path(this_dir) / 'flash_attn' ,
                Path(this_dir) / 'flash_attn' / 'src',
                Path(this_dir)  / 'cutlass' / 'include',
            ],
        ),

        CUDAExtension(
                name="col_linear_lib",
                sources=[
                    "linear/linear_op.cpp",
                    "linear/gemm.cu",
                ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        "--ptxas-options=-v",
                        "-lineinfo"
                        "-lcublas"
                    ]
                    + generator_flag
                    + cc_flag
                ),
            },
            include_dirs=[
                Path(this_dir) / 'linear' ,
            ],
        ),

    ],
    cmdclass={
        'build_ext': BuildExtension
})






