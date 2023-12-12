import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

GibiByte = 1024**3

logger = logging.Logger(__name__)


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
        dtype: The data type for weights and activations.
        tp_size: Tensor parallel size.
        pp_size: Pipeline parallel size.
        max_seq_len: Maximum length of input sentence.
        quant_mode: Quantization mode.
        revision: The specific version(a branch, name, a commit id, or a tag name) of model to use.
        beam_width: The maximum beam width used to initialize KV Cache.
            During generation, the beam width provided as sampling parameter should be less than or equivalent to this value.
        prefill_ratio: A controling ratio for prefill and decoding in running list, we will do a step of prefill
            when the actual value exceeds this ratio.
    """

    model: Union[str, nn.Module]
    tokenizer: str = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    max_batch_size: int = None
    max_output_len: int = 256
    max_input_len: int = 256
    block_size: int = 16
    dtype: Union[str, torch.dtype] = torch.float32
    tp_size: int = 1
    pp_size: int = 1
    max_seq_len: Optional[int] = None
    quant_mode: Optional[str] = None
    revision: Optional[str] = None
    beam_width: int = 1
    # TODO: beam search is not support for now
    prefill_ratio: Optional[float] = 1.2
    # the ratio of prefill sequences to decoding sequences, we do prefill step once the actual value exceeds ratio

    def _init_batch_size(self):
        """
        MAX_BATCH_SIZE is set to acurately utilize the memory of gpu.
        We take a simple method to determine it by GPU memory size, user can still set it manually.
        """
        if self.max_batch_size is not None:
            # already set by user
            return

        device = torch.device("cuda")
        total_mem = torch.cuda.get_device_properties(device).total_memory // GibiByte
        self.max_batch_size = 8

        if 40 < total_mem <= 60:
            self.max_batch_size = 16
        elif 60 < total_mem <= 80:
            self.max_batch_size = 32
        logger.info(
            f"The maximum batch size is automatically set to {self.max_batch_size} as no value is provided by the user."
        )

    def __post_init__(self):
        self._init_batch_size()
        self._verify_args()

    def _verify_args(self):
        if self.tokenizer_mode not in ["auto", "slow"]:
            raise ValueError("Tokenizer mode must be " "either 'auto' or 'slow'," f"but got {self.tokenizer_mode}")
