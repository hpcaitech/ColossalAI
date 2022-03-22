#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from abc import ABC, abstractmethod
from colossalai.logging import get_dist_logger
from torch import Tensor
from typing import Dict

__all__ = ['BaseGradScaler']


class BaseGradScaler(ABC):

    def __init__(self, initial_scale: int, verbose: bool):
        assert initial_scale > 0
        self._scale = torch.cuda.FloatTensor([initial_scale])
        self._verbose = verbose

        if self._verbose:
            self._logger = get_dist_logger()

    @property
    def scale(self) -> Tensor:
        return self._scale

    @property
    def inv_scale(self) -> Tensor:
        return self._scale.double().reciprocal().float()

    def state_dict(self) -> Dict:
        state_dict = dict()
        state_dict['scale'] = self.scale

    def load_state_dict(self, state_dict: Dict) -> None:
        self._scale = state_dict['scale']

    @abstractmethod
    def update(self, overflow: bool) -> None:
        pass

    def log(self, message, *args, **kwargs):
        if self._verbose:
            self._logger.info(message, *args, **kwargs)
