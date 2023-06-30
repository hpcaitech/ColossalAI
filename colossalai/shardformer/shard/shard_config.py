from dataclasses import dataclass

import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.cluster.dist_coordinator import DistCoordinator

__all__ = ['ShardConfig']


@dataclass
class ShardConfig:
    r"""
    The config for sharding the huggingface model

    Args:
        tensor_parallel_process_group (int): The process group for tensor parallelism, defaults to None, which is the global process group.
        enable_fused_normalization (bool): Whether to use fused layernorm, default is False
    """
    tensor_parallel_process_group: int = None
    enable_fused_normalization: bool = False

    # TODO: add support for tensor parallel
    # pipeline_parallel_size: int
    # data_parallel_size: int
    # tensor_parallel_mode: Literal['1d', '2d', '2.5d', '3d']
    # inference_only: bool = True
    # gather_output: bool = True

    def __post_init__(self):
        # get the parallel size
        self.tensor_parallel_size = dist.get_world_size(self.tensor_parallel_process_group)
