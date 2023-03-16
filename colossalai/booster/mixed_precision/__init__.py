from .bf16 import BF16_MixedPrecision
from .fp8 import FP8_MixedPrecision
from .fp16_apex import FP16_Apex_MixedPrecision
from .fp16_torch import FP16_Torch_MixedPrecision
from .mixed_precision_base import MixedPrecision

__all__ = [
    'MixedPrecision', 'mixed_precision_factory', 'FP16_Apex_MixedPrecision', 'FP16_Torch_MixedPrecision',
    'FP32_MixedPrecision', 'BF16_MixedPrecision', 'FP8_MixedPrecision'
]

_mixed_precision_mapping = {
    'fp16': FP16_Torch_MixedPrecision,
    'fp16_apex': FP16_Apex_MixedPrecision,
    'bf16': BF16_MixedPrecision,
    'fp8': FP8_MixedPrecision
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
            f'Mixed precision type {mixed_precision_type} is not supported, support types include {list(_mixed_precision_mapping.keys())}'
        )
