import subprocess

import click
import torch
from torch.utils.cpp_extension import CUDA_HOME


def check_installation():
    cuda_ext_installed = _check_cuda_extension_installed()
    cuda_version, torch_version, torch_cuda_version, cuda_torch_compatibility = _check_cuda_torch()

    click.echo(f"CUDA Version: {cuda_version}")
    click.echo(f"PyTorch Version: {torch_version}")
    click.echo(f"CUDA Version in PyTorch Build: {torch_cuda_version}")
    click.echo(f"PyTorch CUDA Version Match: {cuda_torch_compatibility}")
    click.echo(f"CUDA Extension: {cuda_ext_installed}")


def _check_cuda_extension_installed():
    try:
        import colossalai._C.fused_optim
        is_cuda_extension_installed = u'\u2713'
    except ImportError:
        is_cuda_extension_installed = 'x'
    return is_cuda_extension_installed


def _check_cuda_torch():
    # get cuda version
    if CUDA_HOME is None:
        cuda_version = 'N/A (CUDA_HOME is not set)'
    else:
        raw_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True)
        output = raw_output.split()
        release_idx = output.index("release") + 1
        release = output[release_idx].split(".")
        bare_metal_major = release[0]
        bare_metal_minor = release[1][0]
        cuda_version = f'{bare_metal_major}.{bare_metal_minor}'

    # get torch version
    torch_version = torch.__version__

    # get cuda version in pytorch build
    torch_cuda_major = torch.version.cuda.split(".")[0]
    torch_cuda_minor = torch.version.cuda.split(".")[1]
    torch_cuda_version = f'{torch_cuda_major}.{torch_cuda_minor}'

    # check version compatiblity
    cuda_torch_compatibility = 'x'
    if CUDA_HOME:
        if torch_cuda_major == bare_metal_major:
            if torch_cuda_minor == bare_metal_minor:
                cuda_torch_compatibility = u'\u2713'
            else:
                cuda_torch_compatibility = u'\u2713 (minor version mismatch)'

    return cuda_version, torch_version, torch_cuda_version, cuda_torch_compatibility
