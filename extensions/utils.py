import os
import subprocess
import warnings
from typing import List


def print_rank_0(message: str) -> None:
    """
    Print on only one process to avoid spamming.
    """
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            is_main_rank = True
        else:
            is_main_rank = dist.get_rank() == 0
    except ImportError:
        is_main_rank = True

    if is_main_rank:
        print(message)


def get_cuda_version_in_pytorch() -> List[int]:
    """
    This function returns the CUDA version in the PyTorch build.

    Returns:
        The CUDA version required by PyTorch, in the form of tuple (major, minor).
    """
    import torch

    try:
        torch_cuda_major = torch.version.cuda.split(".")[0]
        torch_cuda_minor = torch.version.cuda.split(".")[1]
    except:
        raise ValueError(
            "[extension] Cannot retrieve the CUDA version in the PyTorch binary given by torch.version.cuda"
        )
    return torch_cuda_major, torch_cuda_minor


def get_cuda_bare_metal_version(cuda_dir) -> List[int]:
    """
    Get the System CUDA version from nvcc.

    Args:
        cuda_dir (str): the directory for CUDA Toolkit.

    Returns:
        The CUDA version required by PyTorch, in the form of tuple (major, minor).
    """
    nvcc_path = os.path.join(cuda_dir, "bin/nvcc")

    if cuda_dir is None:
        raise ValueError(
            f"[extension] The argument cuda_dir is None, but expected to be a string. Please make sure your have exported the environment variable CUDA_HOME correctly."
        )

    # check for nvcc path
    if not os.path.exists(nvcc_path):
        raise FileNotFoundError(
            f"[extension] The nvcc compiler is not found in {nvcc_path}, please make sure you have set the correct value for CUDA_HOME."
        )

    # parse the nvcc -v output to obtain the system cuda version
    try:
        raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
        output = raw_output.split()
        release_idx = output.index("release") + 1
        release = output[release_idx].split(".")
        bare_metal_major = release[0]
        bare_metal_minor = release[1][0]
    except:
        raise ValueError(
            f"[extension] Failed to parse the nvcc output to obtain the system CUDA bare metal version. The output for 'nvcc -v' is \n{raw_output}"
        )

    return bare_metal_major, bare_metal_minor


def check_system_pytorch_cuda_match(cuda_dir):
    bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(cuda_dir)
    torch_cuda_major, torch_cuda_minor = get_cuda_version_in_pytorch()

    if bare_metal_major != torch_cuda_major:
        raise Exception(
            f"[extension] Failed to build PyTorch extension because the detected CUDA version ({bare_metal_major}.{bare_metal_minor}) "
            f"mismatches the version that was used to compile PyTorch ({torch_cuda_major}.{torch_cuda_minor})."
            "Please make sure you have set the CUDA_HOME correctly and installed the correct PyTorch in https://pytorch.org/get-started/locally/ ."
        )

    if bare_metal_minor != torch_cuda_minor:
        warnings.warn(
            f"[extension] The CUDA version on the system ({bare_metal_major}.{bare_metal_minor}) does not match with the version ({torch_cuda_major}.{torch_cuda_minor}) torch was compiled with. "
            "The mismatch is found in the minor version. As the APIs are compatible, we will allow compilation to proceed. "
            "If you encounter any issue when using the built kernel, please try to build it again with fully matched CUDA versions"
        )
    return True


def get_pytorch_version() -> List[int]:
    """
    This functions finds the PyTorch version.

    Returns:
        A tuple of integers in the form of (major, minor, patch).
    """
    import torch

    torch_version = torch.__version__.split("+")[0]
    TORCH_MAJOR = int(torch_version.split(".")[0])
    TORCH_MINOR = int(torch_version.split(".")[1])
    TORCH_PATCH = int(torch_version.split(".")[2], 16)
    return TORCH_MAJOR, TORCH_MINOR, TORCH_PATCH


def check_pytorch_version(min_major_version, min_minor_version) -> bool:
    """
    Compare the current PyTorch version with the minium required version.

    Args:
        min_major_version (int): the minimum major version of PyTorch required
        min_minor_version (int): the minimum minor version of PyTorch required

    Returns:
        A boolean value. The value is True if the current pytorch version is acceptable and False otherwise.
    """
    # get pytorch version
    torch_major, torch_minor, _ = get_pytorch_version()

    # if the
    if torch_major < min_major_version or (torch_major == min_major_version and torch_minor < min_minor_version):
        raise RuntimeError(
            f"[extension] Colossal-AI requires Pytorch {min_major_version}.{min_minor_version} or newer.\n"
            "The latest stable release can be obtained from https://pytorch.org/get-started/locally/"
        )


def check_cuda_availability():
    """
    Check if CUDA is available on the system.

    Returns:
        A boolean value. True if CUDA is available and False otherwise.
    """
    import torch

    return torch.cuda.is_available()


def set_cuda_arch_list(cuda_dir):
    """
    Set default targets when building CUDA extensions without a visible GPU.

    Preserve explicit targets and let PyTorch's extension build backend generate
    architecture flags for both AOT and JIT builds. With visible GPUs and no
    override, PyTorch selects the targets itself.
    """
    cuda_available = check_cuda_availability()
    if cuda_available:
        return True

    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        cuda_version = tuple(int(part) for part in get_cuda_bare_metal_version(cuda_dir))

        # CUDA 13 removed offline compilation for Pascal and Volta.
        arch_list = ["7.5"] if cuda_version >= (13, 0) else ["6.0", "6.1", "6.2", "7.0", "7.5"]
        if cuda_version >= (11, 0):
            arch_list.append("8.0")
        if cuda_version >= (11, 1):
            arch_list.append("8.6")
        if cuda_version >= (11, 8):
            arch_list.extend(["8.9", "9.0"])
        if cuda_version >= (12, 8):
            import torch

            # Both nvcc and PyTorch's architecture parser must recognize these
            # Blackwell targets. Only parse major/minor to allow nightly/RC builds.
            torch_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
            if torch_version >= (2, 7):
                arch_list.extend(["10.0", "12.0"])
            else:
                warnings.warn(
                    "[extension] Omitting native Blackwell targets from the default architecture list: "
                    "PyTorch >= 2.7 is required to parse both 10.0 and 12.0.",
                    stacklevel=2,
                )

        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)

    warnings.warn(
        "\n[extension] PyTorch did not find available GPUs on this system. "
        "This is expected when cross-compiling CUDA extensions.\n"
        f"Building for TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}.\n"
        "Set TORCH_CUDA_ARCH_LIST before building to choose different target architectures. "
        "The targets must be supported by both the CUDA Toolkit and PyTorch.\n",
        stacklevel=2,
    )
    return False


def append_nvcc_threads(nvcc_extra_args: List[str]) -> List[str]:
    """
    This function appends the threads flag to your nvcc args.

    Returns:
        The nvcc compilation flags including the threads flag.
    """
    from torch.utils.cpp_extension import CUDA_HOME

    bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args
