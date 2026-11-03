import warnings
from functools import lru_cache
from importlib.metadata import version
from importlib.util import find_spec
from typing import List

from packaging.version import Version

from .extensions import (
    CpuAdamArmExtension,
    CpuAdamX86Extension,
    FlashAttentionDaoCudaExtension,
    FlashAttentionDaoV3CudaExtension,
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
                    usable_exts.append(ext)

        assert len(usable_exts) != 0, f"No usable kernel found for {self.__class__.__name__} on the current machine."

        for ext in usable_exts:
            ext.assert_compatible()

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


class FlashAttentionDaoLoader(KernelLoader):
    REGISTRY = [FlashAttentionDaoCudaExtension]


class FlashAttentionBackendLoader:
    """Load an explicitly selected attention backend without an execution-time fallback.

    Unlike the legacy priority-based loader, a broken installed extension must surface its
    import/ABI error. Only a missing optional package makes a backend unavailable to auto.
    Callers check request-specific capabilities before using this loader.
    """

    @staticmethod
    def is_available(backend: str) -> bool:
        if backend == "sdpa":
            return True
        package = {"fa2": "flash_attn", "fa3": "flash_attn_3"}[backend]
        if find_spec(package) is None:
            return False
        # FA2 changed non-square causal alignment in 2.1. Earlier releases do
        # not implement the shared lower-right contract. Do not infer a version
        # for an unidentifiable/broken installation.
        return backend != "fa2" or Version(version("flash-attn")) >= Version("2.1")

    @staticmethod
    @lru_cache(maxsize=6)
    def load(backend: str, packed: bool = False):
        extension = {
            "fa2": FlashAttentionDaoCudaExtension,
            "fa3": FlashAttentionDaoV3CudaExtension,
            "sdpa": FlashAttentionSdpaCudaExtension,
        }[backend]()
        if not FlashAttentionBackendLoader.is_available(backend):
            if backend == "fa2" and find_spec("flash_attn") is not None:
                raise NotImplementedError("The explicit FA2 backend requires flash-attn >= 2.1")
            raise ModuleNotFoundError(f"Attention backend {backend!r} requires its optional package to be installed")
        if packed:
            return extension.load_varlen()
        if backend == "sdpa":
            return extension.load(strict=True)
        return extension.load()


class FlashAttentionWithCustomMaskLoader(KernelLoader):
    REGISTRY = [FlashAttentionNpuExtension, FlashAttentionSdpaCudaExtension]


class FlashAttentionForFloatAndCustomMaskLoader(KernelLoader):
    REGISTRY = [FlashAttentionSdpaCudaExtension]
