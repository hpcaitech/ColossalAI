"""
Our config contains various options for inference optimization, it is a unified API that wraps all the configurations for inference.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers.generation import GenerationConfig

from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.inference.utils import can_use_flash_attn2

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
}


class RPC_PARAM(ABC):
    """
    NOTE(lry89757) We use rpyc to transport param between client and server.
    Rpyc only support the type of `POD` in python as the param, so we should take some smart ways to transport the data like tensor or some sophisticated classes.
    Drawing on the logic of `__setstate__`, `__getstate__`, we will let some classes(will be rpc param later) inherit this base class, and rewrite the to_rpc_param and from_rpc_param. We will invoke `to_rpc_param` in client to pass the params and recover the param in server side by `from_rpc_param`.
    """

    @abstractmethod
    def to_rpc_param(self):
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def from_rpc_param():
        return NotImplementedError


@dataclass
class InputMetaData(RPC_PARAM):
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
    batch_token_ids (List[List[int]], optional): input_token_ids + output_token_ids of current batch. Only used for `repetition_penalty`, `no_repeat_ngram_size` in sampler process.
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
    batch_token_ids: Optional[List[List[int]]] = (
        None  # for `repetition_penalty`, `no_repeat_ngram_size` in sampler process
    )

    def to_rpc_param(self) -> Dict[str, any]:
        return {
            "block_tables": self.block_tables.tolist(),
            "sequence_lengths": self.sequence_lengths.tolist(),
            "batch_size": self.batch_size,
            "is_prompts": self.is_prompts,
            "use_cuda_kernel": self.use_cuda_kernel,
            "use_cuda_graph": self.use_cuda_graph,
            "kv_seq_len": self.kv_seq_len,
            "head_dim": self.head_dim,
            "high_precision": self.high_precision,
            "dtype": str(self.dtype).split(".")[-1],
            "use_spec_dec": self.use_spec_dec,
            "num_tokens_to_verify": self.num_tokens_to_verify,
            "batch_token_ids": self.batch_token_ids,
        }

    @staticmethod
    def from_rpc_param(rpc_dict: Dict[str, any]) -> "InputMetaData":
        """
        We intentionally don't use `dict.get` method to ensure we pass the right rpc param, or program will show error message
        """
        from colossalai.accelerator import get_accelerator

        dtype = getattr(torch, rpc_dict["dtype"])
        return InputMetaData(
            block_tables=torch.tensor(
                rpc_dict["block_tables"], dtype=torch.int, device=get_accelerator().get_current_device()
            ),
            sequence_lengths=torch.tensor(
                rpc_dict["sequence_lengths"], dtype=torch.int, device=get_accelerator().get_current_device()
            ),
            batch_size=rpc_dict["batch_size"],
            is_prompts=rpc_dict["is_prompts"],
            use_cuda_kernel=rpc_dict["use_cuda_kernel"],
            use_cuda_graph=rpc_dict["use_cuda_graph"],
            kv_seq_len=rpc_dict["kv_seq_len"],
            head_dim=rpc_dict["head_dim"],
            high_precision=rpc_dict["high_precision"],
            dtype=dtype,
            use_spec_dec=rpc_dict["use_spec_dec"],
            num_tokens_to_verify=rpc_dict["num_tokens_to_verify"],
            batch_token_ids=rpc_dict["batch_token_ids"],
        )

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
class InferenceConfig(RPC_PARAM):
    """The inference configuration.

    Args:
        max_batch_size (int): Maximum batch size, defaults to 8.
        max_output_len (int): Maximum output length, defaults to 256.
        max_input_len (int): Maximum input length, defaults to 256.
        dtype (Union[str, torch.dtype]): The data type for weights and activations.
        kv_cache_dtype (Optional[str]): The data type of kv_cache, defaults to None.
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
        temperature (Optional[float]): Randomness used to control randomization, defaults to 1.0.
        no_repeat_ngram_size (Optional[int]): If no_repeat_ngram_size > 0, the consecutive tokens of ngram size can only appear once in inference sentences.
        repetition_penalty (Optional[float]): The parameter that influences the model's treatment of new tokens in relation to their appearance in the prompt and the generated text. Values greater than 1 incentivize the model to introduce new tokens, whereas values less than 1 incentivize token repetition., defaults to 1.0.
        ignore_eos(bool): Whether to ignore the EOS token and continue generating tokens when encountering the EOS token.
        use_spec_dec (bool): Indicate whether to use speculative decoding, defaults to False.
        max_n_spec_tokens (int): The maximum number of speculating tokens, defaults to None.
        glimpse_large_kv (bool): Whether to use large KV in drafter model, defaults to False.
        block_size (int): The number of blocks in a logical block, defaults to 16.
        tp_size (int): Tensor parallel size, defaults to 1.
        pp_size (int): Pipeline parallel size, defaults to 1.
        micro_batch_size (int): the micro batch size, defaults to 1. Only useful when `pp_size` > 1.
        micro_batch_buffer_size (int): the buffer size for micro batch. Normally, it should be the same as the number of pipeline stages.
        use_cuda_kernel(bool): Whether to use cuda kernel, faster but lose some precision occasionally
        high_precision(Optional[bool]): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, defaults to False.
        use_cuda_graph (bool): Whether to enforce CUDA graph execution. If False, we will disable CUDA graph and always execute the model in eager mode. If True, we will use eager execution in hybrid.
        max_context_len_to_capture (int): max context len that could be captured by CUDA Graph, per sequence
        enable_streamingllm(bool): Whether to use StreamingLLM, the relevant algorithms refer to the paper at https://arxiv.org/pdf/2309.17453 for implementation.
        start_token_size(int): The size of the start tokens, when using StreamingLLM.
        generated_token_size(int): The size of the generated tokens, When using StreamingLLM.
        patched_parallelism_size(int): Patched Parallelism Size, When using Distrifusion
    """

    # NOTE: arrange configs according to their importance and frequency of usage

    # runtime limit
    max_batch_size: int = 8
    max_output_len: int = 256
    max_input_len: int = 256

    # general configs
    dtype: Union[str, torch.dtype] = torch.float16  # use fp16 by default
    kv_cache_dtype: Optional[str] = None

    # generation configs
    prompt_template: Optional[str] = None
    do_sample: bool = False
    beam_width: int = 1  # TODO: beam search is not support for now
    prefill_ratio: Optional[float] = (
        1.2  # the ratio of prefill sequences to decoding sequences, we do prefill step once the actual value exceeds ratio
    )
    pad_input: bool = False
    early_stopping: Optional[bool] = False
    top_k: Optional[int] = 50
    top_p: Optional[float] = 1.0
    temperature: Optional[float] = 1.0
    no_repeat_ngram_size: Optional[int] = 0
    repetition_penalty: Optional[float] = 1.0
    forced_eos_token_id: int = None
    ignore_eos: bool = False

    # speculative decoding configs
    use_spec_dec: bool = False
    max_n_spec_tokens: int = 5
    glimpse_large_kv: bool = False

    # paged attention configs
    block_size: int = 16

    # model parallelism configs
    tp_size: int = 1
    pp_size: int = 1
    micro_batch_size: int = 1
    micro_batch_buffer_size: int = None

    # cuda kernel option
    use_cuda_kernel: bool = False
    high_precision: Optional[bool] = False

    # cuda_graph
    use_cuda_graph: bool = (
        False  # NOTE only when we have the graph for specific decoding batch size can we use the cuda graph for inference
    )
    max_context_len_to_capture: int = 512

    # StreamingLLM (sliding window attention with attention sinks)
    enable_streamingllm: bool = False
    start_token_size: int = 4
    generated_token_size: int = 512

    # Acceleration for Diffusion Model(PipeFusion or Distrifusion)
    patched_parallelism_size: int = 1  # for distrifusion
    # pipeFusion_m_size: int = 1  # for pipefusion
    # pipeFusion_n_size: int = 1  # for pipefusion

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

        if self.kv_cache_dtype:
            assert (
                self.use_cuda_kernel and self.kv_cache_dtype == "fp8"
            ), f"FP8 kv_cache is only supported with use_cuda_kernel open now"
            self.kv_cache_dtype = torch.uint8

        # skip using casting when the data type is float32
        if self.dtype == torch.float32:
            self.high_precision = False

        # check StreamingLLM
        assert (
            self.start_token_size <= self.block_size
        ), f"According to the paper https://arxiv.org/pdf/2309.17453, the start_token_size greater than 4 has little impact on inference performance. Therefore, we assume that the start_token_size should be less or equal than the block_size={self.block_size}, but got {self.start_token_size}."
        assert (
            self.generated_token_size % self.block_size == 0
        ), f"We assume that the generated_token_size should be a multiple of the block_size, got generated_token_size={self.generated_token_size}."
        # Our StreamingLLM implementation (sliding window attention with attention sinks) references https://arxiv.org/pdf/2309.17453 and has been optimized
        # based on our framework's kvcache management mechanism. According to the paper, a start_token_size of 4 is sufficient. Therefore,
        # we assume the start_token_size is less than or equal to the block size. When the start_token_size is smaller than the block size,
        # we fill the first block with the start_token_size and subsequently generated tokens, using these as the "start tokens."
        # Thereafter, we swap out tokens in units of blocks, and always swapping out the second block when the generated tokens exceeded the limit.
        self.start_token_size = self.block_size

        # check Distrifusion
        # TODO(@lry89757) need more detailed check
        if self.patched_parallelism_size > 1:
            # self.use_patched_parallelism = True
            self.tp_size = (
                self.patched_parallelism_size
            )  # this is not a real tp, because some annoying check, so we have to set this to patched_parallelism_size

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
        for type in ["repetition_penalty", "no_repeat_ngram_size", "temperature", "top_k", "top_p"]:
            if hasattr(self, type):
                meta_config[type] = getattr(self, type)
        for type in ["pad_token_id", "bos_token_id", "eos_token_id"]:
            if hasattr(model_config, type):
                meta_config[type] = getattr(model_config, type)

        return GenerationConfig.from_dict(meta_config)

    def to_model_shard_inference_config(self) -> "ModelShardInferenceConfig":
        use_flash_attn = can_use_flash_attn2(self.dtype)
        model_inference_config = ModelShardInferenceConfig(
            dtype=self.dtype,
            use_cuda_kernel=self.use_cuda_kernel,
            use_spec_dec=self.use_spec_dec,
            use_flash_attn=use_flash_attn,
            patched_parallelism_size=self.patched_parallelism_size,
        )
        return model_inference_config

    def to_rpc_param(self) -> dict:
        kwargs = {
            "dtype": str(self.dtype).split(".")[-1],
            "max_n_spec_tokens": self.max_n_spec_tokens,
            "max_batch_size": self.max_batch_size,
            "max_input_len": self.max_input_len,
            "max_output_len": self.max_output_len,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "pad_input": self.pad_input,
            "early_stopping": self.early_stopping,
            "do_sample": self.do_sample,
            "beam_width": self.beam_width,
            "kv_cache_dtype": str(self.kv_cache_dtype).split(".")[-1],
        }
        return kwargs

    @staticmethod
    def from_rpc_param(rpc_dict: dict) -> "InferenceConfig":
        """
        We intentionally don't use `dict.get` method to ensure we pass the right rpc param, or program will show error message
        """
        return InferenceConfig(
            dtype=getattr(torch, rpc_dict["dtype"]),
            max_n_spec_tokens=rpc_dict["max_n_spec_tokens"],
            max_batch_size=rpc_dict["max_batch_size"],
            max_input_len=rpc_dict["max_input_len"],
            max_output_len=rpc_dict["max_output_len"],
            tp_size=rpc_dict["tp_size"],
            pp_size=rpc_dict["pp_size"],
            pad_input=rpc_dict["pad_input"],
            early_stopping=rpc_dict["early_stopping"],
            do_sample=rpc_dict["do_sample"],
            beam_width=rpc_dict["beam_width"],
            kv_cache_dtype=getattr(torch, rpc_dict["kv_cache_dtype"], None),
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "InferenceConfig":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in fields(cls)]
        inference_config_args = {}
        for attr in attrs:
            if attr in config_dict:
                inference_config_args[attr] = config_dict[attr]
            else:
                inference_config_args[attr] = getattr(cls, attr)

        # Set the attributes from the parsed arguments.
        inference_config = cls(**inference_config_args)
        return inference_config


@dataclass
class ModelShardInferenceConfig:
    """
    Configurations used during init of module for inference modeling.

    Args:
        dtype (torch.dtype): The data type for weights and activations.
        use_cuda_kernel (bool): Whether to use cuda kernel, faster but lose some precision occasionally
        use_spec_dec (bool): Indicate whether to use speculative decoding.
        use_flash_attn (bool): Indicate whether to use flash attention.
    """

    dtype: torch.dtype = None
    use_cuda_kernel: bool = False
    use_spec_dec: bool = False
    use_flash_attn: bool = False
    patched_parallelism_size: int = 1  # for diffusion model, Distrifusion Technique


@dataclass
class DiffusionGenerationConfig:
    """
    Param for diffusion model forward
    """

    prompt_2: Optional[Union[str, List[str]]] = None
    prompt_3: Optional[Union[str, List[str]]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: int = None
    timesteps: List[int] = None
    guidance_scale: float = None
    negative_prompt: Optional[Union[str, List[str]]] = (
        None  # NOTE(@lry89757) in pixart default to "", in sd3 default to None
    )
    negative_prompt_2: Optional[Union[str, List[str]]] = None
    negative_prompt_3: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: Optional[int] = None
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    latents: Optional[torch.FloatTensor] = None
    prompt_embeds: Optional[torch.FloatTensor] = None
    negative_prompt_embeds: Optional[torch.FloatTensor] = None
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    output_type: Optional[str] = None  # "pil"
    return_dict: bool = None
    joint_attention_kwargs: Optional[Dict[str, Any]] = None
    clip_skip: Optional[int] = None
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None
    callback_on_step_end_tensor_inputs: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        # NOTE(@lry89757) Only return the dict that not the default value None
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                result[field.name] = value
        return result

    @classmethod
    def from_kwargs(cls, **kwargs) -> "DiffusionGenerationConfig":
        return cls(**kwargs)
