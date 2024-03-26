from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.pipeline.stage_manager import PipelineStageManager

__all__ = ["ShardConfig", "AdvancedPipelineConfig"]


class AdvancedPipelineConfig:
    r"""
    The advanced pipeline config is designed to provide more flexibility for users to customize the pipeline parallelism.
    Refer to https://github.com/hpcaitech/ColossalAI/issues/5509 for more details.

    It provides the following features:
        1. `gradient_checkpointing_ratio`: This is used to control gradient checkpointing more precisely, e.g., set 50% of the layers to use gradient checkpointing.
        2. Customize # layers and # ckpt layers assigned to each stage. This takes precedence over `gradient_checkpointing_ratio`.

    """

    def __init__(
        self,
        gradient_checkpointing_ratio: Optional[float] = None,
        num_stages: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
        num_model_layers: Optional[int] = None,
        num_layers_per_stage: Optional[List[int]] = None,
        num_ckpt_layers_per_stage: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            gradient_checkpointing_ratio (Optional[float]): The ratio of gradient checkpointing. It can only be used in pipeline parallelism. Defaults to None.
            num_stages (Optional[int]): Number of stages in the pipeline. Defaults to None. For sanity check.
            num_model_chunks (Optional[int]): Number of model chunks (1F1B or Interleaved). Defaults to None. For sanity check.
            num_model_layers (Optional[int]): Number of model layers. Defaults to None. For sanity check.
            num_layers_per_stage (Optional[List[int]]): Number of layers for each stage. Defaults to None.
            num_ckpt_layers_per_stage (Optional[List[int]]): Number of checkpointed layers for each stage. Defaults to None.

        Example 1:
            num_stages = 8
            num_layers = 80
            num_model_chunks = 1
            num_layers_per_stage = [9, 9, 9, 10, 11, 10, 11, 11]
            num_ckpt_layers_per_stage = [4, 4, 2, 2, 0, 0, 0, 0]

        Example 2:
            num_stages = 4
            num_layers = 80
            num_model_chunks = 2
            num_layers_per_stage = [9, 9, 9, 10, 11, 10, 11, 11]
            # device 0 holds num_layers_per_stage[0] and num_layers_per_stage[4] layers
            ...

        """
        self.gradient_checkpointing_ratio = gradient_checkpointing_ratio
        self.num_stages = num_stages
        self.num_model_chunks = num_model_chunks
        self.num_model_layers = num_model_layers
        self.num_layers_per_stage = num_layers_per_stage
        self.num_ckpt_layers_per_stage = num_ckpt_layers_per_stage
        self._sanity_check()

    @property
    def enable_gradient_checkpointing_ratio(self) -> bool:
        return self.gradient_checkpointing_ratio is not None

    @property
    def enable_customized_layers_per_stage(self) -> bool:
        return self.num_layers_per_stage is not None

    @property
    def enable_customized_ckpt_layers_per_stage(self) -> bool:
        return self.num_ckpt_layers_per_stage is not None

    def _sanity_check(self):
        if self.gradient_checkpointing_ratio is not None:
            if not (0 <= self.gradient_checkpointing_ratio <= 1):
                raise ValueError("gradient_checkpointing_ratio should be in 0% to 100%")

        if self.num_layers_per_stage is not None:
            assert (
                self.num_stages is not None and self.num_model_chunks is not None and self.num_model_layers is not None
            )
            assert all([0 < num_layers < self.num_model_layers for num_layers in self.num_layers_per_stage])
            assert sum(self.num_layers_per_stage) == self.num_model_layers
            assert len(self.num_layers_per_stage) == self.num_stages * self.num_model_chunks

        if self.num_ckpt_layers_per_stage is not None:
            assert self.num_layers_per_stage is not None
            assert len(self.num_layers_per_stage) == len(self.num_ckpt_layers_per_stage)
            assert all(
                [
                    0 <= num_ckpt_layers <= num_layers
                    for num_ckpt_layers, num_layers in zip(self.num_ckpt_layers_per_stage, self.num_layers_per_stage)
                ]
            )
            self.gradient_checkpointing_ratio = sum(self.num_ckpt_layers_per_stage) / sum(self.num_layers_per_stage)

    def distribute_layers(self, num_layers: int, num_stages: int) -> List[int]:
        assert self.enable_customized_layers_per_stage
        assert num_layers == self.num_model_layers and num_stages == self.num_stages
        return self.num_layers_per_stage

    def get_num_ckpt_layers(self, stage: int, num_layers: int, model_chunk_id: int = 1) -> int:
        if self.enable_customized_layers_per_stage:
            assert stage <= self.num_stages and model_chunk_id <= self.num_model_chunks
            assert num_layers == self.num_layers_per_stage[stage]

        if self.enable_customized_ckpt_layers_per_stage:
            return self.num_ckpt_layers_per_stage[stage]
        elif self.enable_gradient_checkpointing_ratio:
            return int(self.gradient_checkpointing_ratio * num_layers)
        else:
            raise RuntimeError("No checkpointed layers information is provided")


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
        advanced_pipeline_config (Optional[AdvancedPipelineConfig]): The advanced pipeline config for more flexibility in pipeline parallelism. Defaults to None.
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
    parallel_output: bool = True
    advanced_pipeline_config: Optional[AdvancedPipelineConfig] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    # TODO padding vocab
    # make_vocab_size_divisible_by: int = 128
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
