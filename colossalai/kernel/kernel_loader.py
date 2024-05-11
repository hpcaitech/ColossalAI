import warnings
from typing import List

from .extensions import (
    CpuAdamArmExtension,
    CpuAdamX86Extension,
    FlashAttentionDaoCudaExtension,
    FlashAttentionNpuExtension,
    FlashAttentionSdpaCudaExtension,
    FusedOptimizerCudaExtension,
    InferenceOpsCudaExtension,
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
    "InferenceOpsLoader",
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

        if ext_name:
            usable_exts = list(filter(lambda ext: ext.name == ext_name, exts))
        else:
            usable_exts = []
            for ext in exts:
                if ext.is_available():
                    # make sure the machine is compatible during kernel loading
                    ext.assert_compatible()
                    usable_exts.append(ext)

        assert len(usable_exts) != 0, f"No usable kernel found for {self.__class__.__name__} on the current machine."

        if len(usable_exts) > 1:
            # if more than one usable kernel is found, we will try to load the kernel with the highest priority
            usable_exts = sorted(usable_exts, key=lambda ext: ext.priority, reverse=True)
            warnings.warn(
                f"More than one kernel is available, loading the kernel with the highest priority - {usable_exts[0].__class__.__name__}"
            )
        return usable_exts[0].load()


class CPUAdamLoader(KernelLoader):
    REGISTRY = [CpuAdamX86Extension, CpuAdamArmExtension]


class LayerNormLoader(KernelLoader):
    REGISTRY = [LayerNormCudaExtension]


class MoeLoader(KernelLoader):
    REGISTRY = [MoeCudaExtension]


class FusedOptimizerLoader(KernelLoader):
    REGISTRY = [FusedOptimizerCudaExtension]


class InferenceOpsLoader(KernelLoader):
    REGISTRY = [InferenceOpsCudaExtension]


class ScaledMaskedSoftmaxLoader(KernelLoader):
    REGISTRY = [ScaledMaskedSoftmaxCudaExtension]


class ScaledUpperTriangleMaskedSoftmaxLoader(KernelLoader):
    REGISTRY = [ScaledUpperTriangleMaskedSoftmaxCudaExtension]


class FlashAttentionLoader(KernelLoader):
    REGISTRY = [
        FlashAttentionNpuExtension,
        FlashAttentionDaoCudaExtension,
        FlashAttentionSdpaCudaExtension,
    ]


class FlashAttentionWithCustomMaskLoader(KernelLoader):
    REGISTRY = [FlashAttentionNpuExtension, FlashAttentionSdpaCudaExtension]


class FlashAttentionForFloatAndCustomMaskLoader(KernelLoader):
    REGISTRY = [FlashAttentionSdpaCudaExtension]
