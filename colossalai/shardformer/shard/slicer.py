import os
from typing import Dict, Tuple
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .shardconfig import ShardConfig


class Slicer():
    def __init__(
        self, 
        shardconfig: ShardConfig #TODO
    ) -> None:
        self.shardconfig = shardconfig

    def slice_weight_bias(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        dim: int,
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        weight = self.slice_tensor(weight, dim, False)
        bias = self.slice_tensor(bias, dim, True)
        return weight, bias

    def slice_tensor(
        self,
        tensor_in: torch.Tensor,
        dim: int,
        is_bias: bool,
    ) -> torch.Tensor:
        """
        Slice tensor according to the config
        """
        tensor_in = tensor_in[:tensor_in.shape[0]//2]
        return tensor_in