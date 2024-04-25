from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GradientCheckpointConfig:
    gradient_checkpointing_ratio: float = 0.0

    def get_num_ckpt_layers(self, num_layers: int) -> int:
        return int(self.gradient_checkpointing_ratio * num_layers)


@dataclass
class PipelineGradientCheckpointConfig(GradientCheckpointConfig):
    r"""
    The pipeline gradient config is designed to provide more flexibility for users to control gradient checkpoint in pipeline parallelism.
    Combined with PipelineStageManager.set_distribution_config, user can fully control the distribution of layers and checkpointed layers in pipeline parallelism.
    Refer to https://github.com/hpcaitech/ColossalAI/issues/5509 for more details.

    It provides the following features:
        1. `gradient_checkpointing_ratio`: This is used to control gradient checkpointing more precisely, e.g., set 50% of the layers to use gradient checkpointing.
        2. Customize # ckpt layers assigned to each stage. This takes precedence over `gradient_checkpointing_ratio`.

    """

    """
    Args:
        gradient_checkpointing_ratio (Optional[float]): The ratio of gradient checkpointing. It can only be used in pipeline parallelism. Defaults to None.
        num_stages (Optional[int]): Number of stages in the pipeline. Defaults to None. For sanity check.
        num_model_chunks (Optional[int]): Number of model chunks (1F1B or Interleaved). Defaults to None. For sanity check.
        num_model_layers (Optional[int]): Number of model layers. Defaults to None. For sanity check.
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
    num_ckpt_layers_per_stage: Optional[List[int]] = None

    def __post_init__(self):
        if self._enable_customized_ckpt_layers_per_stage:
            assert all([num_ckpt_layers >= 0 for num_ckpt_layers in self.num_ckpt_layers_per_stage])
        elif self._enable_gradient_checkpointing_ratio:
            if not (0 <= self.gradient_checkpointing_ratio <= 1):
                raise ValueError("gradient_checkpointing_ratio should be in 0% to 100%")

    @property
    def _enable_gradient_checkpointing_ratio(self) -> bool:
        return self.gradient_checkpointing_ratio is not None

    @property
    def _enable_customized_ckpt_layers_per_stage(self) -> bool:
        return self.num_ckpt_layers_per_stage is not None

    def get_num_ckpt_layers(
        self, stage: int, num_stages: int, num_layers: int, model_chunk_id: int = 0, num_model_chunks: int = 1
    ) -> int:
        if not self._enable_gradient_checkpointing_ratio and not self._enable_customized_ckpt_layers_per_stage:
            raise RuntimeError("No checkpointed layers information is provided")

        if self._enable_customized_ckpt_layers_per_stage:
            assert len(self.num_ckpt_layers_per_stage) == num_stages * num_model_chunks
            assert stage <= num_stages and model_chunk_id <= num_model_chunks
            num_ckpt_layers = self.num_ckpt_layers_per_stage[stage + model_chunk_id * num_stages]
            assert num_ckpt_layers <= num_layers
            return num_ckpt_layers
        else:
            return int(self.gradient_checkpointing_ratio * num_layers)
