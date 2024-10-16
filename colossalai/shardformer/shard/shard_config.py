import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.pipeline.stage_manager import PipelineStageManager

from .grad_ckpt_config import GradientCheckpointConfig

__all__ = ["ShardConfig"]
SUPPORT_SP_MODE = ["split_gather", "ring", "all_to_all", "ring_attn"]


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
        gradient_checkpoint_config (Optional[GradientCheckpointConfig]): The gradient checkpoint config. Defaults to None.
        enable_all_optimization (bool): Whether to turn on all optimization tools including 'fused normalization', 'flash attention', 'JIT fused operators', 'sequence parallelism' and 'sequence overlap'. Defaults to False.
        fp8_communication (bool, optional): Whether to enable fp8 communication in model parallelism. Defaults to False.
        parallel_output (bool): For TP: whether to use parallelize cross entropy computation along the feature dim.
            For SP: set to True to NOT gather the output along the seq dim.
    """

    tensor_parallel_process_group: Optional[ProcessGroup] = None
    sequence_parallel_process_group: Optional[ProcessGroup] = None
    pipeline_stage_manager: Optional[PipelineStageManager] = None
    enable_tensor_parallelism: bool = True
    enable_all_optimization: bool = False
    enable_fused_normalization: bool = False
    enable_flash_attention: bool = False
    enable_jit_fused: bool = False
    enable_sequence_parallelism: bool = False
    sequence_parallelism_mode: str = None
    enable_sequence_overlap: bool = False
    parallel_output: bool = True
    make_vocab_size_divisible_by: int = 64
    gradient_checkpoint_config: Optional[GradientCheckpointConfig] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # For ring attention
    inner_ring_size: Optional[int] = None
    # for moe related
    moe_dp_group: Optional[ProcessGroup] = None
    ep_group: Optional[ProcessGroup] = None
    fp8_communication: bool = False
    # pipeline_parallel_size: int
    # data_parallel_size: int
    # tensor_parallel_mode: Literal['1d', '2d', '2.5d', '3d']

    @property
    def tensor_parallel_size(self):
        return self._tensor_parallel_size

    @property
    def sequence_parallel_size(self):
        return self._sequence_parallel_size

    def __post_init__(self):
        # turn on all optimization if all_optimization is set to True
        if self.enable_all_optimization:
            self._turn_on_all_optimization()

        if self.enable_sequence_parallelism:
            self.sequence_parallelism_mode = (
                "split_gather" if self.sequence_parallelism_mode is None else self.sequence_parallelism_mode
            )
            assert (
                self.sequence_parallelism_mode in SUPPORT_SP_MODE
            ), f"Sequence parallelism mode {self.sequence_parallelism_mode} is not in the supported list {SUPPORT_SP_MODE}"
            if self.sequence_parallelism_mode in ["split_gather", "ring"]:
                assert (
                    self.enable_tensor_parallelism
                ), f"sequence parallelism mode {self.sequence_parallelism_mode} can only be used when enable_tensor_parallelism is True"
            elif self.sequence_parallelism_mode in ["all_to_all"]:
                # assert (
                #     not self.enable_tensor_parallelism
                # ), f"sequence parallelism mode {self.sequence_parallelism_mode} can only be used when enable_tensor_parallelism is False"
                if self.enable_sequence_overlap:
                    self.enable_sequence_overlap = False
                    warnings.warn(
                        f"The enable_sequence_overlap flag will be ignored in sequence parallelism mode {self.sequence_parallelism_mode}"
                    )
        else:
            if self.sequence_parallelism_mode:
                self.sequence_parallelism_mode = None
                warnings.warn(
                    f"The sequence_parallelism_mode will be ignored when enable_sequence_parallelism is False"
                )
            assert (
                not self.enable_sequence_overlap
            ), f"enable_sequence_overlap can only be set to True when enable_sequence_parallelism is True"

        # get the tensor parallel size
        if not self.enable_tensor_parallelism:
            self._tensor_parallel_size = 1
        else:
            self._tensor_parallel_size = dist.get_world_size(self.tensor_parallel_process_group)

        # get the sequence parallel size
        if not self.enable_sequence_parallelism:
            self._sequence_parallel_size = 1
        else:
            self._sequence_parallel_size = dist.get_world_size(self.sequence_parallel_process_group)

    def _turn_on_all_optimization(self):
        """
        Turn on all optimization.
        """
        # you can add all the optimization flag here
        try:
            from apex.normalization import FusedLayerNorm as ApexFusedLayerNorm  # noqa

            apex_avail = True
        except ImportError:
            apex_avail = False
            warnings.warn("You set enable_all_optimization=True, but apex is not installed.")

        self.enable_fused_normalization = apex_avail
        self.enable_flash_attention = True
        self.enable_jit_fused = True
        # This can cause non-in-place param sharding when used without ZeRO.
        # It may also slow down training when seq len is small. Plz enable manually.
        # self.enable_sequence_parallelism = True
        # self.enable_sequence_overlap = True
