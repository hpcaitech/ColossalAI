from typing import Union

import torch
import torch.nn as nn
from transformers import AutoConfig, PretrainedConfig


class InferConfig:
    """The infer configuration.

    Args:

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

        self.hf_model_config = self._get_hf_model_config()

        def _get_hf_model_config(self) -> PretrainedConfig:
            return AutoConfig.from_pretrained(
                self.model, trust_remote_code=self.trust_remote_code, revision=self.revision
            )

        def get_pp_layer_num(self):
            return self.hf_config.num_hidden_layers // self.pp_size

        def _verify_args(self):
            if self.gpu_utilization_rate > 1.0:
                raise ValueError(
                    f"GPU utilization should be less than 1.0, but is set to {self.gpu_memory_utilization}."
                )
            if self.tokenizer_mode not in ["auto", "slow"]:
                raise ValueError("Tokenizer mode must be " "either 'auto' or 'slow'," f"but got {self.tokenizer_mode}")

            if self.hf_config.num_hidden_layers % self.pp_size != 0:
                raise ValueError(
                    "When using pipeline parallel,"
                    "total number of hidden layers must be divisible by pipeline parallel size."
                    f"Now total number of hidden layers is {self.hf_config.num_hidden_layers},"
                    f"pipeline parallel size is {self.pp_size}"
                )
