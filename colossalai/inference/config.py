from typing import Optional, Union

import torch
import torch.nn as nn


class ColossalInferConfig:
    """The infer configuration.

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

    def __init__(
        self,
        model: Union[str, nn.Module],
        tokenizer: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        max_batch_size: int,
        max_output_len: int,
        max_input_len: int,
        block_size: int,
        gpu_utilization_rate: float,
        dtype: Union[str, torch.dtype],
        tp_size: int = 1,
        pp_size: int = 1,
        max_seq_len: Optional[int] = None,
        quant_mode: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.max_batch_size = max_batch_size
        self.max_output_len = max_output_len
        self.max_input_len = max_input_len
        self.block_size = block_size
        self.gpu_utilization_rate = gpu_utilization_rate
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.quant_mode = quant_mode
        self.revision = revision
        
        self._verify_args()

        def _verify_args(self):
            if self.gpu_utilization_rate > 1.0:
                raise ValueError(
                    f"GPU utilization should be less than 1.0, but is set to {self.gpu_memory_utilization}."
                )
            if self.tokenizer_mode not in ["auto", "slow"]:
                raise ValueError("Tokenizer mode must be " "either 'auto' or 'slow'," f"but got {self.tokenizer_mode}")
