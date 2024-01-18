from typing import List

from .extensions import (
    CpuAdamArmExtension,
    CpuAdamX86Extension,
    FlashAttentionCudaExtension,
    FlashAttentionNpuExtension,
    FusedOptimizerCudaExtension,
    LayerNormCudaExtension,
    MoeCudaExtension,
    ScaledMaskedSoftmaxCudaExtension,
    ScaledUpperTriangleMaskedSoftmaxCudaExtension,
)
from .extensions.base_extension import _Extension

__all__ = [
    "KernelLoader",
    "CPUAdamLoader",
    "LayerNormLoader",
    "MoeLoader",
    "FusedOptimizerLoader",
    "ScaledMaskedSoftmaxLoader",
    "ScaledUpperTriangleMaskedSoftmaxLoader",
]


class KernelLoader:
    """
    An abstract class which offers encapsulation to the kernel loading process.

    Usage:
        kernel_loader = KernelLoader()
        kernel = kernel_loader.load()
    """

    REGISTRY: List[_Extension] = []

    @classmethod
    def register_extension(cls, extension: _Extension):
        """
        This classmethod is an extension point which allows users to register their customized
        kernel implementations to the loader.

        Args:
            extension (_Extension): the extension to be registered.
        """
        cls.REGISTRY.append(extension)

    def load(self, ext_name: str = None):
        """
        Load the kernel according to the current machine.

        Args:
            ext_name (str): the name of the extension to be loaded. If not specified, the loader
                will try to look for an kernel available on the current machine.
        """
        exts = [ext_cls() for ext_cls in self.__class__.REGISTRY]

        # look for exts which can be built/loaded on the current machine
        usable_exts = []
        for ext in exts:
            if ext.name == ext_name:
                # if the user specified the extension name, we will only look for that extension
                usable_exts.append(ext)
                break

            if ext.is_hardware_available():
                # make sure the machine is compatible during kernel loading
                ext.assert_hardware_compatible()
                usable_exts.append(ext)
        assert len(usable_exts) != 0, f"No usable kernel found for {self.__class__.__name__} on the current machine."
        assert (
            len(usable_exts) == 1
        ), f"More than one usable kernel found for {self.__class__.__name__} on the current machine."
        return usable_exts[0].load()


class CPUAdamLoader(KernelLoader):
    REGISTRY = [CpuAdamX86Extension, CpuAdamArmExtension]


class LayerNormLoader(KernelLoader):
    REGISTRY = [LayerNormCudaExtension]


class MoeLoader(KernelLoader):
    REGISTRY = [MoeCudaExtension]


class FusedOptimizerLoader(KernelLoader):
    REGISTRY = [FusedOptimizerCudaExtension]


class ScaledMaskedSoftmaxLoader(KernelLoader):
    REGISTRY = [ScaledMaskedSoftmaxCudaExtension]


class ScaledUpperTriangleMaskedSoftmaxLoader(KernelLoader):
    REGISTRY = [ScaledUpperTriangleMaskedSoftmaxCudaExtension]


class FlashAttentionLoader(KernelLoader):
    REGISTRY = [FlashAttentionNpuExtension, FlashAttentionCudaExtension]
