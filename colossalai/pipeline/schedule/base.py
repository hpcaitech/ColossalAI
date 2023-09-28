from typing import Any, Callable, Iterable, Optional

from torch import Tensor
from torch.nn import Module

from colossalai.interface import OptimizerWrapper
from colossalai.pipeline.stage_manager import PipelineStageManager


class PipelineSchedule:
    def __init__(self, stage_manager: PipelineStageManager) -> None:
        self.stage_manager = stage_manager

    def forward_backward_step(
        self,
        model: Module,
        data_iter: Iterable,
        criterion: Callable[[Any, Any], Tensor],
        optimizer: Optional[OptimizerWrapper] = None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> dict:
        """Forward and backward step for pipeline training.

        Args:
            model (Module): Model to be trained.
            data_iter (Iterable): Data iterator.
            criterion (Callable[[Any, Any], Tensor]): Criterion to be used. It should take two arguments: model outputs and inputs, and returns loss tensor.
            optimizer (OptimizerWrapper, optional): Optimizer to be used. Can be None when only forward is executed. Defaults to None.
            return_loss (bool, optional): Whether to return loss. Defaults to False. Whether to return loss.
            return_outputs (bool, optional): Whether to return model outputs. Defaults to False. Whether to return model outputs.

        Returns:
            dict: A dict with keys: 'loss' and 'outputs'.
        """
        raise NotImplementedError
