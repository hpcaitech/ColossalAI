#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional

import torch

from colossalai.accelerator import get_accelerator

from .base_grad_scaler import BaseGradScaler

__all__ = ["DynamicGradScaler"]


class DynamicGradScaler(BaseGradScaler):
    """A gradient scaler which uses dynamic loss scale

    Args:
        initial_scale (float): the initial loss scale, defaults to 2**16
        growth_factor (float): the multiplication factor for increasing loss scale, defaults to 2
        backoff_factor (float): the multiplication factor for decreasing loss scale, defaults to 0.5
        growth_interval (int): the number of steps to increase loss scale when no overflow occurs, defaults to 1000
        min_scale (float): the minimum loss scale, defaults to None
        max_scale (float): the maximum loss scale, defaults to None
        hysteresis (int):  the number of overflows before decreasing loss scale, defaults to 2
        verbose (bool): whether to log messages, defaults to False
    """

    def __init__(
        self,
        initial_scale: float = 2**16,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        min_scale: Optional[float] = None,
        max_scale: Optional[float] = None,
        hysteresis: int = 2,
        verbose: bool = False,
    ):
        a = get_accelerator()
        a.device_count()
        super().__init__(initial_scale, verbose)
        if min_scale:
            self._min_scale = torch.tensor(
                [min_scale], device=get_accelerator().get_current_device(), dtype=torch.float
            )
        else:
            self._min_scale = None

        if max_scale:
            self._max_scale = torch.tensor(
                [max_scale], device=get_accelerator().get_current_device(), dtype=torch.float
            )
        else:
            self._max_scale = None

        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_step = 0
        self._hysteresis = hysteresis
        self._hysteresis_step = 0
        self._sanity_checks()

    def _sanity_checks(self) -> None:
        """Check if the arguments are correct."""

        if self._min_scale:
            assert self._min_scale > 0, "The minimum gradient scale cannot be zero or negative"
            assert self._min_scale <= self._scale, "The minimum gradient scale cannot be greater than the current scale"
        if self._max_scale:
            assert self._max_scale > 0, "The maximum gradient scale cannot be zero or negative"
            assert self._max_scale >= self._scale, "The maximum gradient scale cannot be smaller than the current scale"
        assert self._growth_factor > 1, "The growth factor cannot be equal or smaller than 1"
        assert 0 < self._backoff_factor < 1, "The backoff factor must be between 0 and 1"
        assert self._hysteresis >= 0, "The hysteresis cannot be negative"

    def update(self, overflow: bool) -> None:
        """Update the loss scale.

        Args:
            overflow (bool): whether overflow occurs
        """
        if overflow:
            self._hysteresis_step += 1
            self._growth_step = 0

            if self._hysteresis_step >= self._hysteresis:
                self._backoff_scale()
                self.log(f"Overflow occurs, the loss scale is adjusted to {self.scale.item()}", ranks=[0])
        else:
            self._growth_step += 1
            if self._growth_step == self._growth_interval:
                self._growth_step = 0
                self._hysteresis_step = 0
                self._grow_scale()
                self.log(
                    f"No overflow for consecutive {self._growth_interval} steps, "
                    f"the loss scale is adjusted to {self.scale.item()}",
                    ranks=[0],
                )

    def _backoff_scale(self) -> None:
        """Decrease the loss scale"""

        self._scale = self._scale * self._backoff_factor
        if self._min_scale:
            self._scale = torch.max(self._scale, self._min_scale)

    def _grow_scale(self) -> None:
        """Increase the loss scale"""

        self._scale = self._scale * self._growth_factor
        if self._max_scale:
            self._scale = torch.min(self._scale, self._max_scale)

    def state_dict(self):
        state_dict = dict()
        state_dict["scale"] = self._scale
        state_dict["growth_factor"] = self._growth_factor
        state_dict["backoff_factor"] = self._backoff_factor
        state_dict["hysteresis"] = self._hysteresis
        return state_dict

    def load_state_dict(self, state_dict):
        self._scale = state_dict["scale"].to(get_accelerator().get_current_device())
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._hysteresis = state_dict["hysteresis"]
