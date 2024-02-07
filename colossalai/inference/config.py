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


_DTYPE_MAPPING = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_ALLOWED_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


_DEFAULT_PROMPT_TEMPLATES = {
    "llama": "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n{input_text}[/INST]",
    "vicuna": "USER: {input_text}\n\nASSISTANT: ",
}


@dataclass
class InferenceConfig:
    """The inference configuration.

    Args:
        micro_batch_size (int): the micro batch size, defaults to 1. Only useful when `pp_size` > 1.
        micro_batch_buffer_size (int): the buffer size for micro batch. Normally, it should be the same as the number of pipeline stages.
        max_batch_size (int): Maximum batch size, defaults to 8.
        max_output_len (int): Maximum output length, defaults to 256.
        max_input_len (int): Maximum input length, defaults to 256.
        block_size (int): The number of blocks in a logical block, defaults to 16.
        dtype (Union[str, torch.dtype]): The data type for weights and activations.
        tp_size (int): Tensor parallel size, defaults to 1.
        pp_size (int): Pipeline parallel size, defaults to 1.
        beam_width (int): The maximum beam width used to initialize KV Cache, defaults to 1.
            During generation, the beam width provided as sampling parameter should be less than or equivalent to this value.
        prefill_ratio (Optional[float]): A controling ratio for prefill and decoding in running list, defaults to 1.2. We will do a step of prefill
            when the actual value exceeds this ratio.
        pad_input: Whether to pad all inputs to the max length.
        quant_mode (Optional[str]): Quantization mode.
        revision (Optional[str]): The specific version(a branch, name, a commit id, or a tag name) of model to use.
        prompt_template (Optional[str]): The prompt template for formatting the input text. Some built-in templates include 'llama' and 'vicuna'. Otherwise, the template should contain '{input_text}' for formatting the input text.
    """

    micro_batch_size: int = 1
    micro_batch_buffer_size: int = None
    max_batch_size: int = 8
    max_output_len: int = 256
    max_input_len: int = 256
    block_size: int = 16
    dtype: Union[str, torch.dtype] = torch.float16  # use fp16 by default
    tp_size: int = 1
    pp_size: int = 1
    # TODO: beam search is not support for now
    beam_width: int = 1
    # the ratio of prefill sequences to decoding sequences, we do prefill step once the actual value exceeds ratio
    prefill_ratio: Optional[float] = 1.2
    pad_input: bool = False
    quant_mode: Optional[str] = None
    revision: Optional[str] = None
    prompt_template: Optional[str] = None

    def __post_init__(self):
        self._verify_config()

    def _verify_config(self) -> None:
        """
        Verify the input config
        """
        # check dtype
        if isinstance(self.dtype, str):
            # convert string dtype to torch dtype
            assert (
                self.dtype in _DTYPE_MAPPING
            ), f"Expected the dtype string argument to be in {list(_DTYPE_MAPPING.keys())} but found an unknown dtype: {self.dtype}"
            self.dtype = _DTYPE_MAPPING[self.dtype]
        assert (
            self.dtype in _ALLOWED_DTYPES
        ), f"Expected dtype to be in {_ALLOWED_DTYPES} but found an unknown dtype: {self.dtype}"

        # check distributed
        assert (
            self.tp_size * self.pp_size == dist.get_world_size()
        ), f"TP size({self.tp_size}) * PP size({self.pp_size}) should be equal to the global world size ({dist.get_world_size()})"

        # check prompt template
        if self.prompt_template is None:
            return

        if self.prompt_template in _DEFAULT_PROMPT_TEMPLATES:
            self.prompt_template = _DEFAULT_PROMPT_TEMPLATES[self.prompt_template]
        else:
            # make sure the template can be formatted with input_text
            assert (
                "{input_text}" in self.prompt_template
            ), "The prompt template should contain '{input_text}' for formatting the input text. For example: 'USER: {input_text}\n\nASSISTANT: '"
