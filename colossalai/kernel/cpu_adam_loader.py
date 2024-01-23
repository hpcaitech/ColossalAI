import platform
from collections import OrderedDict

from .base_kernel_loader import BaseKernelLoader
from .extensions.cpu_adam import ArmCPUAdamExtension, X86CPUAdamExtension


class CPUAdamLoader(BaseKernelLoader):
    """
    CPU Adam Loader

    Usage:
        # init
        cpu_adam = CPUAdamLoader().load()
    """

    def __init__(self):
        super().__init__(
            extension_map=OrderedDict(
                arm=ArmCPUAdamExtension,
                x86=X86CPUAdamExtension,
            ),
            supported_device=["cpu"],
        )

    def fetch_kernel(self):
        if platform.machine() == "x86_64":
            kernel = self._extension_map["x86"]().fetch()
        elif platform.machine() == "aarch64":
            kernel = self._extension_map["arm"]().fetch()
        else:
            raise Exception("not supported")
        return kernel
