#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from .base_grad_scaler import BaseGradScaler

__all__ = ['ConstantGradScaler']


class ConstantGradScaler(BaseGradScaler):

    def __init__(self, initial_scale: int, verbose: bool):
        super().__init__(initial_scale, verbose)
        self.log(f"Constant Gradient Scaler is initialized with scale {self.scale}", ranks=[0])

    def update(self, overflow: bool) -> None:
        # do nothing to maintain the current scale value
        pass
