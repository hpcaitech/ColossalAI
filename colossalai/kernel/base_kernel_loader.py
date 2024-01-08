from abc import ABC, abstractmethod
from typing import Dict, List

from .extensions.base_extension import BaseExtension


class BaseKernelLoader(ABC):
    """
    Usage:
        kernel_loader = KernelLoader()
        kernel = kernel_loader.load()
    """

    def __init__(self, extension_map: Dict[str, BaseExtension], supported_device: List[str]):
        self._extension_map = extension_map
        self._supported_device = supported_device

    def run_checks(self):
        # run supported device check and other possible checks
        pass

    @abstractmethod
    def fetch_kernel(self):
        pass

    def load(self):
        self.run_checks()
        return self.fetch_kernel()
