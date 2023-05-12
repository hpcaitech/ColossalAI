import os
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from dataclasses import dataclass

@dataclass
class ShardConfig:
    """
    The config for sharding the huggingface model for test
    """
    fp16: bool
    num_gpus: int
    rank: int
    backend="nccl"
    verbose: str = 'simple'
    seed: int = None
    require_grad: bool = False
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

class Slicer():
    def __init__(self) -> None:
        pass

    def slice_tensor(
        self,
        tensor_in: torch.Tensor,
        dim: int,
        is_bias: bool,
        dist_config: ShardConfig, # TODO
    ) -> torch.Tensor:
        """
        Slice tensor according to the config
        """
        pass