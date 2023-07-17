from dataclasses import dataclass

import torch.distributed as dist
from torch.distributed import ProcessGroup

__all__ = ['ShardConfig']


@dataclass
class ShardConfig:
    r"""
    The config for sharding the huggingface model

    Args:
        tensor_parallel_process_group (int): The process group for tensor parallelism, defaults to None, which is the global process group.
        enable_tensor_parallelism (bool): Whether to turn on tensor parallelism, default is True.
        enable_fused_normalization (bool): Whether to use fused layernorm, default is False.
        enable_all_optimization (bool): Whether to turn on all optimization, default is False.
    """
    tensor_parallel_process_group: ProcessGroup = None
    enable_tensor_parallelism: bool = True
    enable_fused_normalization: bool = False
    enable_all_optimization: bool = False

    # TODO: add support for tensor parallel
    # pipeline_parallel_size: int
    # data_parallel_size: int
    # tensor_parallel_mode: Literal['1d', '2d', '2.5d', '3d']
    # inference_only: bool = True
    # gather_output: bool = True

    @property
    def tensor_parallel_size(self):
        return self._tensor_parallel_size

    def __post_init__(self):
        if not self.enable_tensor_parallelism:
            self._tensor_parallel_size = 1
        else:
            # get the parallel size
            self._tensor_parallel_size = dist.get_world_size(self.tensor_parallel_process_group)

        # turn on all optimization if all_optimization is set to True
        if self.enable_all_optimization:
            self._turn_on_all_optimization()

    def _turn_on_all_optimization(self):
        """
        Turn on all optimization.
        """
        # you can add all the optimization flag here
        self.enable_fused_normalization = True
