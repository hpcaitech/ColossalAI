from abc import abstractmethod
from typing import Any, Callable, Iterator

import torch

from colossalai.interface import ModelWrapper, OptimizerWrapper

from .plugin_base import Plugin


class PipelinePluginBase(Plugin):

    @abstractmethod
    def execute_pipeline(self,
                         data_iter: Iterator,
                         model: ModelWrapper,
                         criterion: Callable[[Any, Any], torch.Tensor],
                         optimizer: OptimizerWrapper,
                         return_loss: bool = True,
                         return_outputs: bool = False) -> dict:
        pass
