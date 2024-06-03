#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod


class BaseGradientHandler(ABC):
    """A basic helper class to handle all-reduce operations of gradients across different parallel groups
    before optimization.

    Args:
        model (Module): Model where the gradients accumulate.
        optimizer (Optimizer): Optimizer for updating the parameters.
    """

    def __init__(self, model, optimizer):
        self._model = model
        self._optimizer = optimizer

    @abstractmethod
    def handle_gradient(self):
        """A method to accumulate gradients across different parallel groups. Users should
        write their own functions or just use the functions in pre-defined subclasses.
        """
