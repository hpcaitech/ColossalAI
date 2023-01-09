import re
import subprocess
from typing import List


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

def get_cuda_cc_flag() -> List:
    """get_cuda_cc_flag

    cc flag for your GPU arch
    """

    # only import torch when needed
    # this is to avoid importing torch when building on a machine without torch pre-installed
    # one case is to build wheel for pypi release
    import torch
    
    cc_flag = []
    for arch in torch.cuda.get_arch_list():
        res = re.search(r'sm_(\d+)', arch)
        if res:
            arch_cap = res[1]
            if int(arch_cap) >= 60:
                cc_flag.extend(['-gencode', f'arch=compute_{arch_cap},code={arch}'])

    return cc_flag

def append_nvcc_threads(nvcc_extra_args):
    from torch.utils.cpp_extension import CUDA_HOME
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args
