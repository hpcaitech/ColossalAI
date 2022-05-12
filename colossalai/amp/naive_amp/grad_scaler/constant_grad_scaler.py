#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from .base_grad_scaler import BaseGradScaler

__all__ = ['ConstantGradScaler']


class ConstantGradScaler(BaseGradScaler):
    """A gradient scaler which uses constant loss scale

    Args:
        initial_scale (float): the initial loss scale
        verbose (bool): whether to log messages
    """

    def __init__(self, initial_scale: int, verbose: bool):
        super().__init__(initial_scale, verbose)
        self.log(f"Constant Gradient Scaler is initialized with scale {self.scale}", ranks=[0])

    def update(self, overflow: bool) -> None:
        """Do nothing to keep the loss scale constant.

        Args:
            overflow (bool): whether overflow occurs
        """
        pass
