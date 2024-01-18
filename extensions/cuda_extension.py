import os

from .cpp_extension import _CppExtension
from .utils import check_pytorch_version, check_system_pytorch_cuda_match, set_cuda_arch_list

__all__ = ["_CudaExtension"]

# Some constants for installation checks
MIN_PYTORCH_VERSION_MAJOR = 1
MIN_PYTORCH_VERSION_MINOR = 10


class _CudaExtension(_CppExtension):
    def is_hardware_available(self) -> bool:
        # cuda extension can only be built if cuda is availabe
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except:
            cuda_available = False
        return cuda_available

    def assert_hardware_compatible(self) -> None:
        from torch.utils.cpp_extension import CUDA_HOME

        if not CUDA_HOME:
            raise RuntimeError(
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

    def build_jit(self) -> None:
        from torch.utils.cpp_extension import CUDA_HOME

        set_cuda_arch_list(CUDA_HOME)
        return super().build_jit()

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
