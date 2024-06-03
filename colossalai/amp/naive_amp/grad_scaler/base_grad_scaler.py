#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import Tensor

from colossalai.accelerator import get_accelerator
from colossalai.logging import get_dist_logger

__all__ = ["BaseGradScaler"]


class BaseGradScaler(ABC):
    """A base class for the gradient scaler.

    Args:
        initial_scale (float): the initial loss scale
        verbose (bool): whether to log messages
    """

    def __init__(self, initial_scale: float, verbose: bool):
        assert initial_scale > 0
        self._scale = torch.tensor([initial_scale], device=get_accelerator().get_current_device(), dtype=torch.float)
        self._verbose = verbose

        if self._verbose:
            self._logger = get_dist_logger()

    @property
    def scale(self) -> Tensor:
        """Returns the loss scale."""

        return self._scale

    @property
    def inv_scale(self) -> Tensor:
        """Returns the inverse of the loss scale."""

        return self._scale.double().reciprocal().float()

    def state_dict(self) -> Dict:
        """Returns the states of the gradient scaler as a dict object."""

        state_dict = dict()
        state_dict["scale"] = self.scale
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load the states of the gradient scaler from a dict object.

        Args:
            state_dict (dict): the states of the gradient scaler
        """

        self._scale = state_dict["scale"]

    @abstractmethod
    def update(self, overflow: bool) -> None:
        """Update the loss scale.

        Args:
            overflow (bool): whether overflow occurs
        """

    def log(self, message, *args, **kwargs):
        """Log messages.

        Args:
            message (str): the message to log
            *args: positional arguments for :class:`colossalai.logging.DistributedLogger`
            **kwargs: key-word arguments for :class:`colossalai.logging.DistributedLogger`
        """

        if self._verbose:
            self._logger.info(message, *args, **kwargs)
