import torch

from colossalai.context.singleton_meta import SingletonMeta
from colossalai.utils import get_current_device


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
        del self._exp_sums
        del self._max_logits

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
    def exp_sums(self):
        assert self.is_initialized, "Intermediate tensors not initialized yet"
        return self._exp_sums

    @property
    def max_logits(self):
        assert self.is_initialized, "Intermediate tensors not initialized yet"
        return self._max_logits

    def initialize(
        self,
        max_batch_size: int,
        num_attn_heads: int,
        kv_max_split_num: int,
        head_dim: int,
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
        self._exp_sums = torch.empty(
            size=(max_batch_size, num_attn_heads, kv_max_split_num), dtype=dtype, device=device
        )
        self._max_logits = torch.empty(
            size=(max_batch_size, num_attn_heads, kv_max_split_num), dtype=dtype, device=device
        )

        self._tensors_initialized = True
