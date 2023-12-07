from dataclasses import dataclass

import torch


@dataclass
class InferenceConfig:
    """This is only for testing the functionalities of KVCacheManager.
    We are expecting to use a single config class to incorporate all the configs for inference,
    including parallel settings, model configs, and cache configs, etc.

    TODO To be modified as needed.
    """

    # Model configs
    num_kv_heads: int  # num_key_value_heads
    num_attention_heads: int  # num_attention_heads
    head_size: int  # hidden_size / num_attention_heads
    num_layers: int  # num_hidden_layers

    # Cache configs
    # `num_blocks` will be calculated in CacheManager
    block_size: int
    beam_width: int
    max_batch_size: int
    max_input_length: int
    max_output_length: int

    # Tensor-specific configs
    dtype: torch.dtype

    # Parallel configs
    tp_size: int = 1
    world_size: int = 1
