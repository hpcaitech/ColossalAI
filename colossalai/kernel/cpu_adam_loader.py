from .base_kernel_loader import BaseKernelLoader
from .extensions.cpu_adam import ArmCPUAdamExtension, X86CPUAdamExtension


class CPUAdamLoader(BaseKernelLoader):
    def __init__(self):
        super().__init__(
            extension_map=dict(
                arm=ArmCPUAdamExtension,
                x86=X86CPUAdamExtension,
            ),
            supported_device=["cpu"],
        )

    def fetch_kernel(self):
        if self._is_x86_available():
            kernel = self._extension_map["x86"].fetch()
        elif self._is_arm_available():
            kernel = self._extension_map["arm"].fetch()
        else:
            raise Exception("not supported")
        return kernel
