import hashlib
import os
from abc import ABC, abstractmethod
from typing import Callable, Union

__all__ = ["_Extension"]


class _Extension(ABC):
    def __init__(self, name: str, support_aot: bool, support_jit: bool, priority: int = 1):
        self._name = name
        self._support_aot = support_aot
        self._support_jit = support_jit
        self.priority = priority

    @property
    def name(self):
        return self._name

    @property
    def support_aot(self):
        return self._support_aot

    @property
    def support_jit(self):
        return self._support_jit

    @staticmethod
    def get_jit_extension_folder_path():
        """
        Kernels which are compiled during runtime will be stored in the same cache folder for reuse.
        The folder is in the path ~/.cache/colossalai/torch_extensions/<cache-folder>.
        The name of the <cache-folder> follows a common format:
            torch<torch_version_major>.<torch_version_minor>_<device_name><device_version>-<hash>

        The <hash> suffix is the hash value of the path of the `colossalai` file.
        """
        import torch

        import colossalai
        from colossalai.accelerator import get_accelerator

        # get torch version
        torch_version_major = torch.__version__.split(".")[0]
        torch_version_minor = torch.__version__.split(".")[1]

        # get device version
        device_name = get_accelerator().name
        device_version = get_accelerator().get_version()

        # use colossalai's file path as hash
        hash_suffix = hashlib.sha256(colossalai.__file__.encode()).hexdigest()

        # concat
        home_directory = os.path.expanduser("~")
        extension_directory = f".cache/colossalai/torch_extensions/torch{torch_version_major}.{torch_version_minor}_{device_name}-{device_version}-{hash_suffix}"
        cache_directory = os.path.join(home_directory, extension_directory)
        return cache_directory

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the hardware required by the kernel is available.
        """

    @abstractmethod
    def assert_compatible(self) -> None:
        """
        Check if the hardware required by the kernel is compatible.
        """

    @abstractmethod
    def build_aot(self) -> Union["CppExtension", "CUDAExtension"]:
        pass

    @abstractmethod
    def build_jit(self) -> Callable:
        pass

    @abstractmethod
    def load(self) -> Callable:
        pass
