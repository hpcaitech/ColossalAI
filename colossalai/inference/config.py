from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

GibiByte = 1024**3


@dataclass
class InferenceConfig:
    """The inference configuration.

    Args:
        model: Path or nn.Module of this model.
        tokenizer: Path of the tokenizer to use.
        tokenizer_mode: "auto" will use the fast tokenizer if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Whether to trust remote code from huggingface.
        max_batch_size: Maximum batch size.
        max_output_len: Maximum output length.
        max_input_len: Maximum input length.
        block_size: The number of blocks in a logical block.
        gpu_utilization_rate: Maximum GPU memory usage ratio.
        dtype: The data type for weights and activations.
        tp_size: Tensor parallel size.
        pp_size: Pipeline parallel size.
        max_seq_len: Maximum length of input sentence.
        quant_mode: Quantization mode.
        revision: The specific version(a branch, name, a commit id, or a tag name) of model to use.
    """

    model: Union[str, nn.Module]
    tokenizer: str = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    max_batch_size: int = 8
    max_output_len: int = 256
    max_input_len: int = 256
    block_size: int = 16
    gpu_utilization_rate: float = 0.7
    dtype: Union[str, torch.dtype] = torch.float32
    tp_size: int = 1
    pp_size: int = 1
    max_seq_len: Optional[int] = None
    quant_mode: Optional[str] = None
    revision: Optional[str] = None
    ratio: Optional[float] = 1.2
    # the ratio of prefill sequences to decoding sequences, we do prefill step once the actual value exceeds ratio

    def __init_batch_size__(self):
        """
        MAX_BATCH_SIZE is set to acurately utilize the memory of gpu.
        We take a simple method to determine it by GPU memory size, user can still set it manually.
        """
        device = torch.device("cuda")
        total_mem = torch.cuda.get_device_properties(device).total_memory // GibiByte
        self.max_batch_size = 8

        if 40 < total_mem <= 60:
            self.max_batch_size = 16
        elif 60 < total_mem <= 80:
            self.max_batch_size = 32

    def __post_init__(self):
        self._verify_args()

    def _verify_args(self):
        if self.gpu_utilization_rate > 1.0:
            raise ValueError(f"GPU utilization should be less than 1.0, but is set to {self.gpu_memory_utilization}.")
        if self.tokenizer_mode not in ["auto", "slow"]:
            raise ValueError("Tokenizer mode must be " "either 'auto' or 'slow'," f"but got {self.tokenizer_mode}")
