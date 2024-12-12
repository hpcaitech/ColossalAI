import os
import time
from abc import abstractmethod
from pathlib import Path
from typing import List

from .base_extension import _Extension
from .cpp_extension import _CppExtension
from .utils import check_pytorch_version, check_system_pytorch_cuda_match, set_cuda_arch_list

__all__ = ["_CudaExtension"]

# Some constants for installation checks
MIN_PYTORCH_VERSION_MAJOR = 1
MIN_PYTORCH_VERSION_MINOR = 10


class _CudaExtension(_CppExtension):
    @abstractmethod
    def nvcc_flags(self) -> List[str]:
        """
        This function should return a list of nvcc compilation flags for extensions.
        """
        return ["-DCOLOSSAL_WITH_CUDA"]

    def is_available(self) -> bool:
        # cuda extension can only be built if cuda is available
        try:
            import torch

            # torch.cuda.is_available requires a device to exist, allow building with cuda extension on build nodes without a device
            # but where cuda is actually available.
            cuda_available = torch.cuda.is_available() or bool(os.environ.get("FORCE_CUDA", 0))
        except:
            cuda_available = False
        return cuda_available

    def assert_compatible(self) -> None:
        from torch.utils.cpp_extension import CUDA_HOME

        if not CUDA_HOME:
            raise AssertionError(
                "[extension] CUDA_HOME is not found. You need to export CUDA_HOME environment variable or install CUDA Toolkit first in order to build/load CUDA extensions"
            )
        check_system_pytorch_cuda_match(CUDA_HOME)
        check_pytorch_version(MIN_PYTORCH_VERSION_MAJOR, MIN_PYTORCH_VERSION_MINOR)

    def get_cuda_home_include(self):
        """
        return include path inside the cuda home.
        """
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            raise RuntimeError("CUDA_HOME is None, please set CUDA_HOME to compile C++/CUDA kernels in ColossalAI.")
        cuda_include = os.path.join(CUDA_HOME, "include")
        return cuda_include

    def include_dirs(self) -> List[str]:
        """
        This function should return a list of include files for extensions.
        """
        return super().include_dirs() + [self.get_cuda_home_include()]

    def build_jit(self) -> None:
        from torch.utils.cpp_extension import CUDA_HOME, load

        set_cuda_arch_list(CUDA_HOME)

        # get build dir
        build_directory = _Extension.get_jit_extension_folder_path()
        build_directory = Path(build_directory)
        build_directory.mkdir(parents=True, exist_ok=True)

        # check if the kernel has been built
        compiled_before = False
        kernel_file_path = build_directory.joinpath(f"{self.name}.so")
        if kernel_file_path.exists():
            compiled_before = True

        # load the kernel
        if compiled_before:
            print(f"[extension] Loading the JIT-built {self.name} kernel during runtime now")
        else:
            print(f"[extension] Compiling the JIT {self.name} kernel during runtime now")

        build_start = time.time()
        op_kernel = load(
            name=self.name,
            sources=self.strip_empty_entries(self.sources_files()),
            extra_include_paths=self.strip_empty_entries(self.include_dirs()),
            extra_cflags=self.cxx_flags(),
            extra_cuda_cflags=self.nvcc_flags(),
            extra_ldflags=[],
            build_directory=str(build_directory),
        )
        build_duration = time.time() - build_start

        if compiled_before:
            print(f"[extension] Time taken to load {self.name} op: {build_duration} seconds")
        else:
            print(f"[extension] Time taken to compile {self.name} op: {build_duration} seconds")

        return op_kernel

    def build_aot(self) -> "CUDAExtension":
        from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

        set_cuda_arch_list(CUDA_HOME)
        return CUDAExtension(
            name=self.prebuilt_import_path,
            sources=self.strip_empty_entries(self.sources_files()),
            include_dirs=self.strip_empty_entries(self.include_dirs()),
            extra_compile_args={
                "cxx": self.strip_empty_entries(self.cxx_flags()),
                "nvcc": self.strip_empty_entries(self.nvcc_flags()),
            },
        )
