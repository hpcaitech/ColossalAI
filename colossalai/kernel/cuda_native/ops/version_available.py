import os
import subprocess
import warnings

import pkg_resources


def triton_cuda_check():
    cuda_home = os.getenv("CUDA_HOME", default="/usr/local/cuda")
    cuda_version = subprocess.check_output([os.path.join(cuda_home, "bin/nvcc"), "--version"]).decode().strip()
    cuda_version = cuda_version.split('release ')[1]
    cuda_version = cuda_version.split(',')[0]
    cuda_version = cuda_version.split('.')
    if len(cuda_version) == 2 and \
        (int(cuda_version[0]) == 11 and int(cuda_version[1]) >= 4) or \
        int(cuda_version[0]) > 11:
        return True
    return False


def get_package_version(package_name):
    try:
        package_version = pkg_resources.get_distribution(package_name).version
        return package_version
    except pkg_resources.DistributionNotFound:
        return None


def triton_available():
    has_triton = False
    package_name = 'xformers'
    if get_package_version(package_name):
        has_triton = True
    else:
        warnings.warn('please install triton from https://github.com/openai/triton')
        has_triton = False
    if triton_cuda_check():
        has_triton = True
    else:
        warnings.warn("triton requires cuda >= 11.4")
        has_triton = False
    return has_triton


def flash_attn_2_available():
    has_flash_attn_2 = False
    package_name = 'flash_attn'
    if get_package_version(package_name):
        has_flash_attn_2 = True
    else:
        warnings.warn('please install flash_attn from https://github.com/HazyResearch/flash-attention')
        has_flash_attn_2 = False

    return has_flash_attn_2


def mem_eff_attn_available():
    has_mem_eff_attn = False
    package_name = 'xformers'
    if get_package_version(package_name):
        has_mem_eff_attn = True
    else:
        warnings.warn('please install xformers from https://github.com/facebookresearch/xformers')
        has_mem_eff_attn = False
    return has_mem_eff_attn


HAS_TRITON = triton_available()
HAS_FLASH_ATTN = flash_attn_2_available()
HAS_MEM_EFF_ATTN = mem_eff_attn_available()

if not HAS_TRITON and not HAS_FLASH_ATTN and not HAS_MEM_EFF_ATTN:
    raise Exception("flash attention can not support!")
print(HAS_TRITON)
print(HAS_FLASH_ATTN)
print(HAS_MEM_EFF_ATTN)
