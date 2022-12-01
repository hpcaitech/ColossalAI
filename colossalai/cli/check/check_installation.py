import subprocess

import click
import torch
from torch.utils.cpp_extension import CUDA_HOME

import colossalai


def check_installation():
    cuda_ext_installed = _check_cuda_extension_installed()
    cuda_version, torch_version, torch_cuda_version = _check_cuda_torch()
    colossalai_verison, torch_version_required, cuda_version_required = _parse_colossalai_version()

    cuda_compatibility = _get_compatibility_string([cuda_version, torch_cuda_version, cuda_version_required])
    torch_compatibility = _get_compatibility_string([torch_version, torch_version_required])

    click.echo(f'#### Installation Report ####\n')
    click.echo(f"Colossal-AI version: {colossalai_verison}")
    click.echo(f'----------------------------')
    click.echo(f"PyTorch Version: {torch_version}")
    click.echo(f"PyTorch Version required by Colossal-AI: {torch_version_required}")
    click.echo(f'PyTorch version match: {torch_compatibility}')
    click.echo(f'----------------------------')
    click.echo(f"System CUDA Version: {cuda_version}")
    click.echo(f"CUDA Version required by PyTorch: {torch_cuda_version}")
    click.echo(f"CUDA Version required by Colossal-AI: {cuda_version_required}")
    click.echo(f"CUDA Version Match: {cuda_compatibility}")
    click.echo(f'----------------------------')
    click.echo(f"CUDA Extension: {cuda_ext_installed}")


def _get_compatibility_string(versions):

    # split version into [major, minor, patch]
    versions = [version.split('.') for version in versions]

    for version in versions:
        if len(version) == 2:
            # x means unknown
            version.append('x')

    for idx, version_values in enumerate(zip(*versions)):
        equal = len(set(version_values)) == 1

        if idx in [0, 1] and not equal:
            # if the major/minor versions do not match
            # return a cross
            return 'x'
        elif idx == 1:
            # if the minor versions match
            # return a tick
            return u'\u2713'
        else:
            continue


def _parse_colossalai_version():
    colossalai_verison = colossalai.__version__.split('+')[0]
    torch_version_required = colossalai.__version__.split('torch')[1].split('cu')[0]
    cuda_version_required = colossalai.__version__.split('cu')[1]
    return colossalai_verison, torch_version_required, cuda_version_required


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
    torch_version = torch.__version__.split('+')[0]

    # get cuda version in pytorch build
    torch_cuda_major = torch.version.cuda.split(".")[0]
    torch_cuda_minor = torch.version.cuda.split(".")[1]
    torch_cuda_version = f'{torch_cuda_major}.{torch_cuda_minor}'

    return cuda_version, torch_version, torch_cuda_version
