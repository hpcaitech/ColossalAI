from .bf16 import BF16MixedPrecision
from .fp8 import FP8MixedPrecision
from .fp16_apex import FP16ApexMixedPrecision
from .fp16_naive import FP16NaiveMixedPrecision
from .fp16_torch import FP16TorchMixedPrecision
from .mixed_precision_base import MixedPrecision

__all__ = [
    "MixedPrecision",
    "mixed_precision_factory",
    "FP16_Apex_MixedPrecision",
    "FP16_Torch_MixedPrecision",
    "FP32_MixedPrecision",
    "BF16_MixedPrecision",
    "FP8_MixedPrecision",
    "FP16NaiveMixedPrecision",
]

_mixed_precision_mapping = {
    "fp16": FP16TorchMixedPrecision,
    "fp16_apex": FP16ApexMixedPrecision,
    "fp16_naive": FP16NaiveMixedPrecision,
    "bf16": BF16MixedPrecision,
    "fp8": FP8MixedPrecision,
}


def mixed_precision_factory(mixed_precision_type: str) -> MixedPrecision:
    """
    Factory method to create mixed precision object

    Args:
        mixed_precision_type (str): mixed precision type, including None, 'fp16', 'fp16_apex', 'bf16', and 'fp8'.
    """

    if mixed_precision_type in _mixed_precision_mapping:
        return _mixed_precision_mapping[mixed_precision_type]()
    else:
        raise ValueError(
            f"Mixed precision type {mixed_precision_type} is not supported, support types include {list(_mixed_precision_mapping.keys())}"
        )
