"""
Our config contains various options for inference optimization, it is a unified API that wraps all the configurations for inference.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed as dist
from transformers.generation import GenerationConfig

from colossalai.inference.flash_decoding_utils import FDIntermTensors

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
    "baichuan": " <reserved_106> {input_text} <reserved_107> ",
    "vicuna": "A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user input. USER: {input_text}\nASSISTANT: ",
    "bloom": "Assume you are a helpful robot. Please help react to my question or auto complete my prompt."
    # "bloom": "[INST] <<SYS>>\nYou are an intelligent and comprehensive assistant. Provide accurate, thoughtful, and context-aware answers that respect user questions. Avoid content that is harmful, misleading, or unethical. Prioritize safety and fairness in all responses. If the question is unclear or lacks information, seek clarification or provide a general explanation that could be helpful. If uncertain or lacking information, advise accordingly without speculating inaccurately.\n<</SYS>>\n{input_text}[/INST]",
}


@dataclass
class InputMetaData:
    """The input info for a single step

    Args:
    block_tables (torch.Tensor, optional): Sequences' BlockTables Defaults to None.
    sequence_lengths (torch.Tensor): A tensor containing sequence lengths.
    fd_inter_tensor (torch.Tensor, optional): A tensor representing intermediate data for flash decoding. Defaults to None.
    batch_size (int, optional): The current batch size. Defaults to 64.
    is_prompts (bool, optional): Indicates whether prefill or decoding. Defaults to False(decoding).
    use_cuda_kernel(bool): Whether to use cuda kernel, faster but lose some precision occasionally
    use_cuda_graph (bool, optional): Indicates whether to use the CUDA graph. Defaults to False.
    kv_seq_len (int, optional): Key-value sequence length. Defaults to 512.
    head_dim (int, optional): Head dimension. Defaults to 32.
    high_precision(bool, optional): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, Defaults to False.
    dtype (torch.dtype, optional): The computation type of tensor, Defaults to torch.float32.
    use_spec_dec (bool): Indicate whether to use speculative decoding.
    num_tokens_to_verify (int): The number of tokens to verify in speculative decoding. Only valid when `use_spec_dec` is set to True.
    """

    block_tables: torch.Tensor = None
    sequence_lengths: torch.Tensor = None
    fd_inter_tensor: FDIntermTensors = None
    batch_size: int = 64  # current_batch_size
    is_prompts: bool = False
    use_cuda_kernel: bool = False
    use_cuda_graph: bool = False
    kv_seq_len: int = 512
    head_dim: int = 32
    high_precision: bool = False
    dtype: torch.dtype = torch.float32
    use_spec_dec: bool = False
    num_tokens_to_verify: int = 0

    def __repr__(self) -> str:
        return (
            f"InputMetaData(block_tables={self.block_tables}, "
            f"sequence_lengths={self.sequence_lengths}, "
            f"fd_inter_tensor={self.fd_inter_tensor}, "
            f"batch_size={self.batch_size}, "
            f"is_prompts={self.is_prompts}, "
            f"use_cuda_kernel={self.use_cuda_kernel}, "
            f"use_cuda_graph={self.use_cuda_graph}, "
            f"kv_seq_len={self.kv_seq_len}, "
            f"use_spec_dec={self.use_spec_dec}, "
            f"num_tokens_to_verify={self.num_tokens_to_verify})"
        )


@dataclass
class InferenceConfig:
    """The inference configuration.

    Args:
        max_batch_size (int): Maximum batch size, defaults to 8.
        max_output_len (int): Maximum output length, defaults to 256.
        max_input_len (int): Maximum input length, defaults to 256.
        dtype (Union[str, torch.dtype]): The data type for weights and activations.
        prompt_template (Optional[str]): The prompt template for generation, defaults to None.
        do_sample (bool): Whether to use sampling for generation, defaults to False.
        beam_width (int): The maximum beam width used to initialize KV Cache, defaults to 1.
            During generation, the beam width provided as sampling parameter should be less than or equivalent to this value.
        prefill_ratio (Optional[float]): A controling ratio for prefill and decoding in running list, defaults to 1.2. We will do a step of prefill
            when the actual value exceeds this ratio.
        pad_input: Whether to pad all inputs to the max length.
        early_stopping (Optional[bool]): Whether to stop the generation when all beam hypotheses have finished or not, defaults to False.
        top_k (Optional[int]): The number of highest probability vocabulary tokens to keep for top-k-filtering, defaults to None.
        top_p (Optional[float]): The cumulative probability threshold for retaining tokens with a total probability above it, defaults to None.
        min_p (Optional[float]): The minimum probability to keep for top-p filtering, defaults to None.
        n_spec_tokens (int): The maximum number of speculating tokens, defaults to None.
        glimpse_large_kv (bool): Whether to use large KV in drafter model, defaults to False.
        block_size (int): The number of blocks in a logical block, defaults to 16.
        tp_size (int): Tensor parallel size, defaults to 1.
        pp_size (int): Pipeline parallel size, defaults to 1.
        micro_batch_size (int): the micro batch size, defaults to 1. Only useful when `pp_size` > 1.
        micro_batch_buffer_size (int): the buffer size for micro batch. Normally, it should be the same as the number of pipeline stages.
        use_cuda_kernel(bool): Whether to use cuda kernel, faster but lose some precision occasionally
        use_cuda_graph (bool): Whether to enforce CUDA graph execution. If False, we will disable CUDA graph and always execute the model in eager mode. If True, we will use eager execution in hybrid.
        max_context_len_to_capture (int): max context len that could be captured by CUDA Graph, per sequence
        high_precision(Optional[bool]): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, defaults to False.
    """

    # NOTE: arrange configs according to their importance and frequency of usage

    # runtime limit
    max_batch_size: int = 8
    max_output_len: int = 256
    max_input_len: int = 256

    # general configs
    dtype: Union[str, torch.dtype] = torch.float16  # use fp16 by default

    # generation configs
    prompt_template: Optional[str] = None
    do_sample: bool = False
    beam_width: int = 1  # TODO: beam search is not support for now
    prefill_ratio: Optional[
        float
    ] = 1.2  # the ratio of prefill sequences to decoding sequences, we do prefill step once the actual value exceeds ratio
    pad_input: bool = False
    early_stopping: Optional[bool] = False
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None

    # speculative decoding configs
    max_n_spec_tokens: int = 5
    glimpse_large_kv: bool = False

    # paged attention configs
    block_size: int = 16

    # model parallelism configs
    tp_size: int = 1
    pp_size: int = 1
    micro_batch_size: int = 1
    micro_batch_buffer_size: int = None
    high_precision: Optional[bool] = False

    # cuda kernel option
    use_cuda_kernel: bool = False

    # cuda_graph
    use_cuda_graph: bool = False  # NOTE only when we have the graph for specific decoding batch size can we use the cuda graph for inference
    max_context_len_to_capture: int = 512

    def __post_init__(self):
        self.max_context_len_to_capture = self.max_input_len + self.max_output_len
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

        # skip using casting when the data type is float32
        if self.dtype == torch.float32:
            self.high_precision = False

        # check distributed
        assert (not torch.distributed.is_initialized() and self.tp_size * self.pp_size == 1) or (
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

    def to_generation_config(self, model_config) -> GenerationConfig:
        meta_config = {
            "max_length": self.max_input_len + self.max_output_len,
            "max_new_tokens": self.max_output_len,
            "early_stopping": self.early_stopping,
            "do_sample": self.do_sample,
            "num_beams": self.beam_width,
        }
        for type in ["top_k", "top_p", "min_p"]:
            if hasattr(self, type):
                meta_config[type] = getattr(self, type)
        for type in ["pad_token_id", "bos_token_id", "eos_token_id"]:
            if hasattr(model_config, type):
                meta_config[type] = getattr(model_config, type)

        return GenerationConfig.from_dict(meta_config)
