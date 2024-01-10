"""
Our config contains various options for inference optimization, it is a unified API that wraps all the configurations for inference.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed as dist

GibiByte = 1024**3

logger = logging.Logger(__name__)


@dataclass
class InferenceConfig:
    """The inference configuration.

    Args:
        micro_batch_size (int): the micro batch size. Only useful when `pp_size` > 1.
        micro_batch_buffer_size (int): the buffer size for micro batch. Normally, it should be the same as the number of pipeline stages.
        max_batch_size (int): Maximum batch size.
        max_output_len (int): Maximum output length.
        max_input_len (int): Maximum input length.
        block_size (int): The number of blocks in a logical block.
        dtype (Union[str, torch.dtype]): The data type for weights and activations.
        tp_size (int): Tensor parallel size.
        pp_size (int): Pipeline parallel size.
        beam_width (int): The maximum beam width used to initialize KV Cache.
            During generation, the beam width provided as sampling parameter should be less than or equivalent to this value.
        prefill_ratio (Optional[float]): A controling ratio for prefill and decoding in running list, we will do a step of prefill
            when the actual value exceeds this ratio.
        quant_mode (Optional[str]): Quantization mode.
        revision (Optional[str]): The specific version(a branch, name, a commit id, or a tag name) of model to use.
    """

    micro_batch_size: int = 1
    micro_batch_buffer_size: int = None
    max_batch_size: int = 8
    max_output_len: int = 256
    max_input_len: int = 256
    block_size: int = 16
    dtype: Union[str, torch.dtype] = torch.float32
    tp_size: int = 1
    pp_size: int = 1
    # TODO: beam search is not support for now
    beam_width: int = 1
    # the ratio of prefill sequences to decoding sequences, we do prefill step once the actual value exceeds ratio
    prefill_ratio: Optional[float] = 1.2
    quant_mode: Optional[str] = None
    revision: Optional[str] = None

    def __post_init__(self):
        self._init_batch_size()
        self._verify_config()

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

    def _verify_config(self) -> None:
        """
        Verify the input config
        """
        assert (
            self.tp_size * self.pp_size == dist.get_world_size()
        ), f"TP size({self.tp_size}) * PP size({self.pp_size}) should be equal to the global world size ({dist.get_world_size()})"
        assert self.dtype in [
            "fp16",
            "fp32",
            "bf16",
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ], f"dtype should be one of 'fp16', 'fp32', 'bf16', torch.float32, torch.float16, torch.bfloat16, but got {self.dtype}."
        assert self.quant_mode in [
            "smoothquant",
            "gptq",
            None,
        ], f"quant should be one of 'smoothquant', 'gptq', but got {self.quant_mode}."
