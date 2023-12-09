from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.pipeline.stage_manager import PipelineStageManager

__all__ = ["ShardConfig"]


@dataclass
class ShardConfig:
    r"""
    The config for sharding the huggingface model

    Args:
        tensor_parallel_process_group (Optional[ProcessGroup]): The process group of tensor parallelism, it's necessary when using tensor parallel. Defaults to None, which is the global process group.
        pipeline_stage_manager (Optional[PipelineStageManager]): If using pipeline parallelism, it's necessary to specify a pipeline stage manager for inter-process communication in pipeline parallelism. Defaults to None, which means not using pipeline parallelism.
        enable_tensor_parallelism (bool): Whether to use tensor parallelism. Defaults to True.
        enable_fused_normalization (bool): Whether to use fused layernorm. Defaults to False.
        enable_flash_attention (bool, optional): Whether to switch on flash attention. Defaults to False.
        enable_jit_fused (bool, optional): Whether to switch on JIT fused operators. Defaults to False.
        enable_sequence_parallelism (bool): Whether to turn on sequence parallelism, which partitions non-tensor-parallel regions along the sequence dimension. Defaults to False.
        enable_sequence_overlap (bool): Whether to turn on sequence overlap, which overlap the computation and communication in sequence parallelism. It can only be used when enable_sequence_parallelism is True. Defaults to False.
        enable_all_optimization (bool): Whether to turn on all optimization tools including 'fused normalization', 'flash attention', 'JIT fused operators', 'sequence parallelism' and 'sequence overlap'. Defaults to False.
    """
    tensor_parallel_process_group: Optional[ProcessGroup] = None
    pipeline_stage_manager: Optional[PipelineStageManager] = None
    enable_tensor_parallelism: bool = True
    enable_fused_normalization: bool = False
    enable_flash_attention: bool = False
    enable_jit_fused: bool = False
    enable_all_optimization: bool = False
    enable_sequence_parallelism: bool = False
    enable_sequence_overlap: bool = False
    test_seq_parallelism: bool = False
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    # pipeline_parallel_size: int
    # data_parallel_size: int
    # tensor_parallel_mode: Literal['1d', '2d', '2.5d', '3d']

    @property
    def tensor_parallel_size(self):
        return self._tensor_parallel_size

    def __post_init__(self):
        if not self.enable_tensor_parallelism and self.enable_sequence_parallelism:
            raise ValueError(
                "enable_sequence_parallelism can only be set to True when enable_tensor_parallelism is True"
            )
        if not self.enable_sequence_parallelism and self.enable_sequence_overlap:
            raise ValueError("enable_sequence_overlap can only be set to True when enable_sequence_parallelism is True")
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
        self.enable_flash_attention = True
        self.enable_jit_fused = True
        self.enable_sequence_parallelism = True
        self.enable_sequence_overlap = True

    def _infer(self):
        """
        Set default params for inference.
        """
        # assert self.pipeline_stage_manager is None, "pipeline parallelism is not supported in inference for now"
