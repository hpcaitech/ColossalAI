# adapted from Hugging Face accelerate/utils/dataclasses.py

import warnings
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class BnbQuantizationConfig:
    """
    A plugin to enable BitsAndBytes 4bit and 8bit quantization
    """

    load_in_8bit: bool = field(default=False, metadata={"help": "enable 8bit quantization."})

    llm_int8_threshold: float = field(
        default=6.0, metadata={"help": "value of the outliner threshold. only relevant when load_in_8bit=True"}
    )

    load_in_4bit: bool = field(default=False, metadata={"help": "enable 4bit quantization."})

    bnb_4bit_quant_type: str = field(
        default="fp4",
        metadata={
            "help": "set the quantization data type in the `bnb.nn.Linear4Bit` layers. Options are {'fp4','np4'}."
        },
    )

    bnb_4bit_use_double_quant: bool = field(
        default=False,
        metadata={
            "help": "enable nested quantization where the quantization constants from the first quantization are quantized again."
        },
    )

    bnb_4bit_compute_dtype: bool = field(
        default="fp16",
        metadata={
            "help": "This sets the computational type which might be different than the input time. For example, inputs might be "
            "fp32, but computation can be set to bf16 for speedups. Options are {'fp32','fp16','bf16'}."
        },
    )

    torch_dtype: torch.dtype = field(
        default=None,
        metadata={
            "help": "this sets the dtype of the remaining non quantized layers. `bitsandbytes` library suggests to set the value"
            "to `torch.float16` for 8 bit model and use the same dtype as the compute dtype for 4 bit model "
        },
    )

    skip_modules: List[str] = field(
        default=None,
        metadata={
            "help": "an explicit list of the modules that we don't quantize. The dtype of these modules will be `torch_dtype`."
        },
    )

    keep_in_fp32_modules: List[str] = field(
        default=None,
        metadata={"help": "an explicit list of the modules that we don't quantize. We keep them in `torch.float32`."},
    )

    def __post_init__(self):
        if isinstance(self.bnb_4bit_compute_dtype, str):
            if self.bnb_4bit_compute_dtype == "fp32":
                self.bnb_4bit_compute_dtype = torch.float32
            elif self.bnb_4bit_compute_dtype == "fp16":
                self.bnb_4bit_compute_dtype = torch.float16
            elif self.bnb_4bit_compute_dtype == "bf16":
                self.bnb_4bit_compute_dtype = torch.bfloat16
            else:
                raise ValueError(
                    f"bnb_4bit_compute_dtype must be in ['fp32','fp16','bf16'] but found {self.bnb_4bit_compute_dtype}"
                )
        elif not isinstance(self.bnb_4bit_compute_dtype, torch.dtype):
            raise ValueError("bnb_4bit_compute_dtype must be a string or a torch.dtype")

        if self.skip_modules is not None and not isinstance(self.skip_modules, list):
            raise ValueError("skip_modules must be a list of strings")

        if self.keep_in_fp32_modules is not None and not isinstance(self.keep_in_fp32_modules, list):
            raise ValueError("keep_in_fp_32_modules must be a list of strings")

        if self.load_in_4bit:
            self.target_dtype = "int4"

        if self.load_in_8bit:
            self.target_dtype = torch.int8

        if self.load_in_4bit and self.llm_int8_threshold != 6.0:
            warnings.warn("llm_int8_threshold can only be used for model loaded in 8bit")

        if isinstance(self.torch_dtype, str):
            if self.torch_dtype == "fp32":
                self.torch_dtype = torch.float32
            elif self.torch_dtype == "fp16":
                self.torch_dtype = torch.float16
            elif self.torch_dtype == "bf16":
                self.torch_dtype = torch.bfloat16
            else:
                raise ValueError(f"torch_dtype must be in ['fp32','fp16','bf16'] but found {self.torch_dtype}")

        if self.load_in_8bit and self.torch_dtype is None:
            self.torch_dtype = torch.float16

        if self.load_in_4bit and self.torch_dtype is None:
            self.torch_dtype = self.bnb_4bit_compute_dtype

        if not isinstance(self.torch_dtype, torch.dtype):
            raise ValueError("torch_dtype must be a torch.dtype")
