import warnings
from dataclasses import dataclass, field
from functools import lru_cache, partial
from importlib.metadata import version
from importlib.util import find_spec
from typing import Callable, List, Optional, Tuple, Type

import torch
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


def _flash_attention_support(q, v, attention_mask, dropout_p, is_causal, causal_alignment, non_square, *, hopper=False):
    # ROCm reuses torch.cuda, but its architecture numbers are not NVIDIA SMs.
    # These adapters are validated for NVIDIA; ROCm needs its own capability policy.
    if torch.version.hip is not None:
        return "requires NVIDIA CUDA in this adapter; ROCm is not supported"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return "requires float16 or bfloat16 inputs"
    if attention_mask is not None:
        return "does not support an arbitrary attention_mask"
    if q.shape[-1] != v.shape[-1] or q.shape[-1] > 256 or q.shape[-1] % 8 != 0:
        return "requires equal Q/K/V head dimensions divisible by 8 and at most 256"
    if not hopper and q.shape[-1] > 192:
        return "supports head dimensions at most 192 in the explicit training adapter"
    if is_causal and non_square and causal_alignment == "upper_left":
        return "does not support non-square upper-left causal attention"
    if hopper and dropout_p != 0:
        return "does not support nonzero dropout"
    major, minor = torch.cuda.get_device_capability(q.device)
    if hopper and (major, minor) != (9, 0):
        return "requires a Hopper (SM90) GPU in this adapter"
    if not hopper and major < 8:
        return "requires an Ampere or newer GPU"
    return None


@dataclass(frozen=True)
class _AttentionBackend:
    extension: Type[_Extension]
    device_types: Optional[Tuple[str, ...]] = None
    capability: Optional[Callable] = None
    package: Optional[str] = None
    minimum_version: Optional[Tuple[str, str]] = None
    load_kwargs: dict = field(default_factory=dict)
    host_lengths: bool = False


class FlashAttentionBackendLoader:
    """Load an explicitly selected attention backend without an execution-time fallback.

    Unlike the legacy priority-based loader, a broken installed extension must surface its
    import/ABI error. Only a missing optional package makes a backend unavailable to auto.
    REGISTRY insertion order defines auto priority within a device type. Generic
    backends (device_types=None) come last. Adapters accept causal_alignment and
    obey True=allowed masks; native conventions are converted inside the adapter.
    Register before first use; clear load's cache if replacing an existing entry.
    """

    REGISTRY = {
        "fa3": _AttentionBackend(
            FlashAttentionDaoV3CudaExtension, ("cuda",), partial(_flash_attention_support, hopper=True), "flash_attn_3"
        ),
        "fa2": _AttentionBackend(
            FlashAttentionDaoCudaExtension, ("cuda",), _flash_attention_support, "flash_attn", ("flash-attn", "2.1")
        ),
        "sdpa": _AttentionBackend(FlashAttentionSdpaCudaExtension, load_kwargs={"strict": True}, host_lengths=True),
    }

    @classmethod
    def get_backend(cls, backend):
        if not isinstance(backend, str) or backend not in cls.REGISTRY:
            raise ValueError(f"Unknown attention backend {backend!r}; expected one of {tuple(cls.REGISTRY)}")
        return cls.REGISTRY[backend]

    @classmethod
    def candidates(cls, device_type):
        specific = [
            name for name, spec in cls.REGISTRY.items() if spec.device_types and device_type in spec.device_types
        ]
        generic = [name for name, spec in cls.REGISTRY.items() if spec.device_types is None]
        return specific + generic

    @staticmethod
    def is_available(backend: str) -> bool:
        spec = FlashAttentionBackendLoader.get_backend(backend)
        if spec.package is not None and find_spec(spec.package) is None:
            return False
        # FA2 changed non-square causal alignment in 2.1. Earlier releases do
        # not implement the shared lower-right contract. Do not infer a version
        # for an unidentifiable/broken installation.
        if spec.minimum_version is not None:
            distribution, minimum = spec.minimum_version
            return Version(version(distribution)) >= Version(minimum)
        return True

    @staticmethod
    @lru_cache(maxsize=None)
    def load(backend: str, packed: bool = False):
        spec = FlashAttentionBackendLoader.get_backend(backend)
        if not FlashAttentionBackendLoader.is_available(backend):
            if spec.minimum_version is not None and find_spec(spec.package) is not None:
                distribution, minimum = spec.minimum_version
                raise NotImplementedError(f"Attention backend {backend!r} requires {distribution} >= {minimum}")
            raise ModuleNotFoundError(f"Attention backend {backend!r} requires its optional package to be installed")
        extension = spec.extension()
        if packed:
            return extension.load_varlen()
        return extension.load(**spec.load_kwargs)


class FlashAttentionWithCustomMaskLoader(KernelLoader):
    REGISTRY = [FlashAttentionNpuExtension, FlashAttentionSdpaCudaExtension]


class FlashAttentionForFloatAndCustomMaskLoader(KernelLoader):
    REGISTRY = [FlashAttentionSdpaCudaExtension]
