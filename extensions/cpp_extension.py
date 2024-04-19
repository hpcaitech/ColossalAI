import importlib
import os
import time
from abc import abstractmethod
from pathlib import Path
from typing import List

from .base_extension import _Extension

__all__ = ["_CppExtension"]


class _CppExtension(_Extension):
    def __init__(self, name: str, priority: int = 1):
        super().__init__(name, support_aot=True, support_jit=True, priority=priority)

        # we store the op as an attribute to avoid repeated building and loading
        self.cached_op = None

        # build-related variables
        self.prebuilt_module_path = "colossalai._C"
        self.prebuilt_import_path = f"{self.prebuilt_module_path}.{self.name}"
        self.version_dependent_macros = ["-DVERSION_GE_1_1", "-DVERSION_GE_1_3", "-DVERSION_GE_1_5"]

    def csrc_abs_path(self, path):
        return os.path.join(self.relative_to_abs_path("csrc"), path)

    def pybind_abs_path(self, path):
        return os.path.join(self.relative_to_abs_path("pybind"), path)

    def relative_to_abs_path(self, code_path: str) -> str:
        """
        This function takes in a path relative to the colossalai root directory and return the absolute path.
        """

        # get the current file path
        # iteratively check the parent directory
        # if the parent directory is "extensions", then the current file path is the root directory
        # otherwise, the current file path is inside the root directory
        current_file_path = Path(__file__)
        while True:
            if current_file_path.name == "extensions":
                break
            else:
                current_file_path = current_file_path.parent
        extension_module_path = current_file_path
        code_abs_path = extension_module_path.joinpath(code_path)
        return str(code_abs_path)

    # functions must be overrided over
    def strip_empty_entries(self, args):
        """
        Drop any empty strings from the list of compile and link flags
        """
        return [x for x in args if len(x) > 0]

    def import_op(self):
        """
        This function will import the op module by its string name.
        """
        return importlib.import_module(self.prebuilt_import_path)

    def build_aot(self) -> "CppExtension":
        from torch.utils.cpp_extension import CppExtension

        return CppExtension(
            name=self.prebuilt_import_path,
            sources=self.strip_empty_entries(self.sources_files()),
            include_dirs=self.strip_empty_entries(self.include_dirs()),
            extra_compile_args=self.strip_empty_entries(self.cxx_flags()),
        )

    def build_jit(self) -> None:
        from torch.utils.cpp_extension import load

        build_directory = _Extension.get_jit_extension_folder_path()
        build_directory = Path(build_directory)
        build_directory.mkdir(parents=True, exist_ok=True)

        # check if the kernel has been built
        compiled_before = False
        kernel_file_path = build_directory.joinpath(f"{self.name}.o")
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
            extra_ldflags=[],
            build_directory=str(build_directory),
        )
        build_duration = time.time() - build_start

        if compiled_before:
            print(f"[extension] Time taken to load {self.name} op: {build_duration} seconds")
        else:
            print(f"[extension] Time taken to compile {self.name} op: {build_duration} seconds")

        return op_kernel

    # functions must be overrided begin
    @abstractmethod
    def sources_files(self) -> List[str]:
        """
        This function should return a list of source files for extensions.
        """

    @abstractmethod
    def include_dirs(self) -> List[str]:
        """
        This function should return a list of include files for extensions.
        """
        return [self.csrc_abs_path("")]

    @abstractmethod
    def cxx_flags(self) -> List[str]:
        """
        This function should return a list of cxx compilation flags for extensions.
        """

    def load(self):
        try:
            op_kernel = self.import_op()
        except (ImportError, ModuleNotFoundError):
            # if import error occurs, it means that the kernel is not pre-built
            # so we build it jit
            op_kernel = self.build_jit()

        return op_kernel
