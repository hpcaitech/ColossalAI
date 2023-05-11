import os
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from .shardmodel import ShardConfig

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