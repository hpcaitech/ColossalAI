import subprocess

import click
import torch
from torch.utils.cpp_extension import CUDA_HOME

import colossalai


def to_click_output(val):
    # installation check output to understandable symbols for readability
    VAL_TO_SYMBOL = {True: "\u2713", False: "x", None: "N/A"}

    if val in VAL_TO_SYMBOL:
        return VAL_TO_SYMBOL[val]
    else:
        return val


def check_installation():
    """
    This function will check the installation of colossalai, specifically, the version compatibility of
    colossalai, pytorch and cuda.

    Example:
    ```text
    ```

    Returns: A table of installation information.
    """
    found_aot_cuda_ext = _check_aot_built_cuda_extension_installed()
    cuda_version = _check_cuda_version()
    torch_version, torch_cuda_version = _check_torch_version()
    colossalai_version, prebuilt_torch_version_required, prebuilt_cuda_version_required = _parse_colossalai_version()

    # if cuda_version is None, that means either
    # CUDA_HOME is not found, thus cannot compare the version compatibility
    if not cuda_version:
        sys_torch_cuda_compatibility = None
    else:
        sys_torch_cuda_compatibility = _is_compatible([cuda_version, torch_cuda_version])

    # if cuda_version or cuda_version_required is None, that means either
    # CUDA_HOME is not found or AOT compilation is not enabled
    # thus, there is no need to compare the version compatibility at all
    if not cuda_version or not prebuilt_cuda_version_required:
        sys_colossalai_cuda_compatibility = None
    else:
        sys_colossalai_cuda_compatibility = _is_compatible([cuda_version, prebuilt_cuda_version_required])

    # if torch_version_required is None, that means AOT compilation is not enabled
    # thus there is no need to compare the versions
    if prebuilt_torch_version_required is None:
        torch_compatibility = None
    else:
        torch_compatibility = _is_compatible([torch_version, prebuilt_torch_version_required])

    click.echo(f"#### Installation Report ####")
    click.echo(f"\n------------ Environment ------------")
    click.echo(f"Colossal-AI version: {to_click_output(colossalai_version)}")
    click.echo(f"PyTorch version: {to_click_output(torch_version)}")
    click.echo(f"System CUDA version: {to_click_output(cuda_version)}")
    click.echo(f"CUDA version required by PyTorch: {to_click_output(torch_cuda_version)}")
    click.echo("")
    click.echo(f"Note:")
    click.echo(f"1. The table above checks the versions of the libraries/tools in the current environment")
    click.echo(f"2. If the System CUDA version is N/A, you can set the CUDA_HOME environment variable to locate it")
    click.echo(
        f"3. If the CUDA version required by PyTorch is N/A, you probably did not install a CUDA-compatible PyTorch. This value is give by torch.version.cuda and you can go to https://pytorch.org/get-started/locally/ to download the correct version."
    )

    click.echo(f"\n------------ CUDA Extensions AOT Compilation ------------")
    click.echo(f"Found AOT CUDA Extension: {to_click_output(found_aot_cuda_ext)}")
    click.echo(f"PyTorch version used for AOT compilation: {to_click_output(prebuilt_torch_version_required)}")
    click.echo(f"CUDA version used for AOT compilation: {to_click_output(prebuilt_cuda_version_required)}")
    click.echo("")
    click.echo(f"Note:")
    click.echo(
        f"1. AOT (ahead-of-time) compilation of the CUDA kernels occurs during installation when the environment variable BUILD_EXT=1 is set"
    )
    click.echo(f"2. If AOT compilation is not enabled, stay calm as the CUDA kernels can still be built during runtime")

    click.echo(f"\n------------ Compatibility ------------")
    click.echo(f"PyTorch version match: {to_click_output(torch_compatibility)}")
    click.echo(f"System and PyTorch CUDA version match: {to_click_output(sys_torch_cuda_compatibility)}")
    click.echo(f"System and Colossal-AI CUDA version match: {to_click_output(sys_colossalai_cuda_compatibility)}")
    click.echo(f"")
    click.echo(f"Note:")
    click.echo(f"1. The table above checks the version compatibility of the libraries/tools in the current environment")
    click.echo(
        f"   - PyTorch version mismatch: whether the PyTorch version in the current environment is compatible with the PyTorch version used for AOT compilation"
    )
    click.echo(
        f"   - System and PyTorch CUDA version match: whether the CUDA version in the current environment is compatible with the CUDA version required by PyTorch"
    )
    click.echo(
        f"   - System and Colossal-AI CUDA version match: whether the CUDA version in the current environment is compatible with the CUDA version used for AOT compilation"
    )


def _is_compatible(versions):
    """
    Compare the list of versions and return whether they are compatible.
    """
    if None in versions:
        return False

    # split version into [major, minor, patch]
    versions = [version.split(".") for version in versions]

    for version in versions:
        if len(version) == 2:
            # x means unknown
            version.append("x")

    for idx, version_values in enumerate(zip(*versions)):
        equal = len(set(version_values)) == 1

        if idx in [0, 1] and not equal:
            return False
        elif idx == 1:
            return True
        else:
            continue


def _parse_colossalai_version():
    """
    Get the Colossal-AI version information.

    Returns:
        colossalai_version: Colossal-AI version.
        torch_version_for_aot_build: PyTorch version used for AOT compilation of CUDA kernels.
        cuda_version_for_aot_build: CUDA version used for AOT compilation of CUDA kernels.
    """
    # colossalai version can be in two formats
    # 1. X.X.X+torchX.XXcuXX.X (when colossalai is installed with CUDA extensions)
    # 2. X.X.X (when colossalai is not installed with CUDA extensions)
    # where X represents an integer.
    colossalai_version = colossalai.__version__.split("+")[0]

    try:
        torch_version_for_aot_build = colossalai.__version__.split("torch")[1].split("cu")[0]
        cuda_version_for_aot_build = colossalai.__version__.split("cu")[1]
    except:
        torch_version_for_aot_build = None
        cuda_version_for_aot_build = None
    return colossalai_version, torch_version_for_aot_build, cuda_version_for_aot_build


def _check_aot_built_cuda_extension_installed():
    """
    According to `op_builder/README.md`, the CUDA extension can be built with either
    AOT (ahead-of-time) or JIT (just-in-time) compilation.
    AOT compilation will build CUDA extensions to `colossalai._C` during installation.
    JIT (just-in-time) compilation will build CUDA extensions to `~/.cache/colossalai/torch_extensions` during runtime.
    """
    try:
        found_aot_cuda_ext = True
    except ImportError:
        found_aot_cuda_ext = False
    return found_aot_cuda_ext


def _check_torch_version():
    """
    Get the PyTorch version information.

    Returns:
        torch_version: PyTorch version.
        torch_cuda_version: CUDA version required by PyTorch.
    """
    # get torch version
    # torch version can be of two formats
    # - 1.13.1+cu113
    # - 1.13.1.devxxx
    torch_version = torch.__version__.split("+")[0]
    torch_version = ".".join(torch_version.split(".")[:3])

    # get cuda version in pytorch build
    try:
        torch_cuda_major = torch.version.cuda.split(".")[0]
        torch_cuda_minor = torch.version.cuda.split(".")[1]
        torch_cuda_version = f"{torch_cuda_major}.{torch_cuda_minor}"
    except:
        torch_cuda_version = None

    return torch_version, torch_cuda_version


def _check_cuda_version():
    """
    Get the CUDA version information.

    Returns:
        cuda_version: CUDA version found on the system.
    """

    # get cuda version
    if CUDA_HOME is None:
        cuda_version = CUDA_HOME
    else:
        try:
            raw_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True)
            output = raw_output.split()
            release_idx = output.index("release") + 1
            release = output[release_idx].split(".")
            bare_metal_major = release[0]
            bare_metal_minor = release[1][0]
            cuda_version = f"{bare_metal_major}.{bare_metal_minor}"
        except:
            cuda_version = None
    return cuda_version
