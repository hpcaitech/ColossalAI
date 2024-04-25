import math

import torch

from colossalai.context.singleton_meta import SingletonMeta
from colossalai.utils import get_current_device


# alibi slopes calculation adapted from https://github.com/huggingface/transformers/blob/v4.36.0/src/transformers/models/bloom/modeling_bloom.py#L57
def get_alibi_slopes(num_heads: int, device: torch.device) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=torch.float32, device=device)
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32, device=device)
    slopes = torch.pow(base, powers)
    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=torch.float32, device=device
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=torch.int32, device=device)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


class FDIntermTensors(metaclass=SingletonMeta):
    """Singleton class to hold tensors used for storing intermediate values in flash-decoding.
    For now, it holds intermediate output and logsumexp (which will be used in reduction step along kv)
    """

    def __init__(self):
        self._tensors_initialized = False

    def _reset(self):
        self._tensors_initialized = False
        del self._mid_output
        del self._mid_output_lse

    @property
    def is_initialized(self):
        return self._tensors_initialized

    @property
    def mid_output(self):
        assert self.is_initialized, "Intermediate tensors not initialized yet"
        return self._mid_output

    @property
    def mid_output_lse(self):
        assert self.is_initialized, "Intermediate tensors not initialized yet"
        return self._mid_output_lse

    @property
    def alibi_slopes(self):
        assert self.is_initialized, "Intermediate tensors not initialized yet"
        return self._alibi_slopes

    def initialize(
        self,
        max_batch_size: int,
        num_attn_heads: int,
        kv_max_split_num: int,
        head_dim: int,
        alibi_attn: bool,
        dtype: torch.dtype = torch.float32,
        device: torch.device = get_current_device(),
    ) -> None:
        """Initialize tensors.

        Args:
            max_batch_size (int): The maximum batch size over all the model forward.
                This could be greater than the batch size in attention forward func when using dynamic batch size.
            num_attn_heads (int)): Number of attention heads.
            kv_max_split_num (int): The maximum number of blocks splitted on kv in flash-decoding algorithm.
                **The maximum length/size of blocks splitted on kv should be the kv cache block size.**
            head_dim (int): Head dimension.
            alibi_attn (bool): Whether to use alibi flash attention.
            dtype (torch.dtype, optional): Data type to be assigned to intermediate tensors.
            device (torch.device, optional): Device used to initialize intermediate tensors.
        """
        assert not self.is_initialized, "Intermediate tensors used for Flash-Decoding have been initialized."

        self._mid_output = torch.empty(
            size=(max_batch_size, num_attn_heads, kv_max_split_num, head_dim), dtype=dtype, device=device
        )
        self._mid_output_lse = torch.empty(
            size=(max_batch_size, num_attn_heads, kv_max_split_num), dtype=dtype, device=device
        )

        if alibi_attn:
            self._alibi_slopes = get_alibi_slopes(num_attn_heads, device)
        else:
            self._alibi_slopes = None

        self._tensors_initialized = True
