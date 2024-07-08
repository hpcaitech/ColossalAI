"""
Utils for model inference
"""

import math
import os
import re
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline
from torch import nn

from colossalai.logging import get_dist_logger
from colossalai.testing import free_port

logger = get_dist_logger(__name__)


def init_to_get_rotary(self, base=10000, use_elem=False):
    """
    This function initializes the rotary positional embedding, it is compatible for all models and is called in ShardFormer
    Args:
        self : Model that holds the rotary positional embedding
        base : calculation arg
        use_elem : activated when using chatglm-based models
    """
    self.config.head_dim_ = self.config.hidden_size // self.config.num_attention_heads
    if not hasattr(self.config, "rope_scaling"):
        rope_scaling_factor = 1.0
    else:
        rope_scaling_factor = self.config.rope_scaling.factor if self.config.rope_scaling is not None else 1.0

    if hasattr(self.config, "max_sequence_length"):
        max_seq_len = self.config.max_sequence_length
    elif hasattr(self.config, "max_position_embeddings"):
        max_seq_len = self.config.max_position_embeddings * rope_scaling_factor
    else:
        max_seq_len = 2048 * rope_scaling_factor
    base = float(base)

    # NTK  ref: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    ntk_alpha = os.environ.get("INFER_NTK_ALPHA", None)

    if ntk_alpha is not None:
        ntk_alpha = float(ntk_alpha)
        assert ntk_alpha >= 1, "NTK alpha must be greater than or equal to 1"
        if ntk_alpha > 1:
            print(f"Note: NTK enabled, alpha set to {ntk_alpha}")
        max_seq_len *= ntk_alpha
        base = base * (ntk_alpha ** (self.head_dim_ / (self.head_dim_ - 2)))  # Base change formula

    n_elem = self.config.head_dim_
    if use_elem:
        n_elem //= 2

    inv_freq = 1.0 / (base ** (torch.arange(0, n_elem, 2, device="cpu", dtype=torch.float32) / n_elem))
    t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
    freqs = torch.outer(t, inv_freq)

    self._cos_cached = torch.cos(freqs).to(self.dtype).cuda()
    self._sin_cached = torch.sin(freqs).to(self.dtype).cuda()


def has_index_file(checkpoint_path: str) -> Tuple[bool, Optional[Path]]:
    """
    Check whether the checkpoint has an index file.

    Args:
        checkpoint_path (str): path to the checkpoint.

    Returns:
        Tuple[bool, Optional[Path]]: a tuple of (has_index_file, index_file_path)
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        # check if it is .index.json
        reg = re.compile("(.*?).index((\..*)?).json")
        if reg.fullmatch(checkpoint_path.name) is not None:
            return True, checkpoint_path
        else:
            return False, None
    elif checkpoint_path.is_dir():
        index_files = list(checkpoint_path.glob("*.index.*json"))

        for index_file in index_files:
            if "safetensors" in index_file.__str__():
                return True, index_file.__str__()  # return the safetensors file first

        if len(index_files) == 1:
            return True, index_files[0]
        else:
            assert (
                len(index_files) == 1
            ), f"Expected to find one .index.json file in {checkpoint_path}, but found {len(index_files)}"
            return False, None
    else:
        raise RuntimeError(f"Invalid checkpoint path {checkpoint_path}. Expected a file or a directory.")


def get_model_size(model: nn.Module):
    """Calculates the total size of the model weights (including biases) in bytes.
    Args:
        model: The PyTorch model to analyze.
    Returns:
        The total size of the model weights in bytes.
    """
    total_size = 0
    for key, param in model.named_parameters():
        total_size += param.element_size() * param.numel()
    return total_size / (1024**3)


def find_available_ports(num: int):
    try:
        free_ports = [free_port() for i in range(num)]
    except OSError as e:
        print(f"An OS error occurred: {e}")
        raise RuntimeError("Error finding available ports")
    return free_ports


def get_alibi_slopes(num_heads: int, device: torch.device) -> torch.Tensor:
    """
    Alibi slopes calculation adapted from https://github.com/huggingface/transformers/blob/v4.36.0/src/transformers/models/bloom/modeling_bloom.py#L57

    Args:
        num_heads (int): The number of attention heads.
        device (torch.device): The device to use.

    Returns:
        torch.Tensor: The Alibi slopes.
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=torch.float32, device=device)
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32, device=device)
    slopes = torch.pow(base, powers)
    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=torch.float32, device=device
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=torch.int32, device=device)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


def can_use_flash_attn2(dtype: torch.dtype) -> bool:
    """
    Check flash attention2 availability.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        return False

    try:
        from flash_attn import flash_attn_varlen_func  # noqa

        return True
    except ImportError:
        logger.warning(f"flash_attn2 has not been installed yet, we will use triton flash attn instead.")
        return False


class ModelType(Enum):
    DIFFUSION_MODEL = "Diffusion Model"
    LLM = "Large Language Model (LLM)"
    UNKNOWN = "Unknown Model Type"


def get_model_type(model_or_path: Union[nn.Module, str, DiffusionPipeline]):
    if isinstance(model_or_path, DiffusionPipeline):
        return ModelType.DIFFUSION_MODEL
    elif isinstance(model_or_path, nn.Module):
        return ModelType.LLM
    elif isinstance(model_or_path, str):
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(model_or_path, trust_remote_code=True)
            return ModelType.LLM
        except:
            """
            model type is not `ModelType.LLM`
            """

        try:
            DiffusionPipeline.load_config(model_or_path)
            return ModelType.DIFFUSION_MODEL
        except:
            """
            model type is not `ModelType.DIFFUSION_MODEL`
            """
    else:
        return ModelType.UNKNOWN
