#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from colossalai.builder import build_gradient_handler
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger
from colossalai.nn import (ZeroRedundancyOptimizer_Level_2,
                           ZeroRedundancyOptimizer_Level_3)
from .schedule import BaseSchedule


class Engine:
    """Basic engine class for training and evaluation. It runs a specific process method 
    :meth:`step` which is based on the given :attr:`schedule` over each batch of a dataset.
    It controls a iteration in training.

    :param model: The neural network model
    :param optimizer: Optimizer for updating the parameters
    :param step_schedule: Running schedule in :meth:`step`
    :param gradient_accumulation: Steps of gradient accumulation
    :param gradient_clipping: The norm of gradient clipping
    :type model: Module
    :type optimizer: Optimizer
    :type step_schedule: BaseSchedule, optional
    :type gradient_accumulation: int, optional
    :type gradient_clipping: float, optional
    """

    def __init__(self,
                 model: Module,
                 optimizer: Optimizer,
                 criterion: _Loss,
                 step_schedule: BaseSchedule,
                 gradient_handlers: list = None,
                 gradient_accumulation: int = 1,
                 gradient_clipping: float = 0.0,
                 ):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._schedule = step_schedule

        # schedule initialize
        self._schedule.initialize(model, optimizer)

        # state
        self.training = True  # default

        # gradient accumulation
        assert gradient_accumulation > 0, 'gradient accumulation size must be larger than 0'
        self._grad_accum_size = gradient_accumulation
        self._grad_clip = gradient_clipping
        self._logger = get_global_dist_logger()

        # build gradient handler
        self._gradient_handlers = []

        if gradient_handlers is not None:
            assert isinstance(gradient_handlers, list), \
                f'argument gradient_handler_cfg expected type list, ' \
                f'but got type {type(gradient_handlers)}'
        elif isinstance(optimizer, (ZeroRedundancyOptimizer_Level_2,
                                    ZeroRedundancyOptimizer_Level_3)):
            gradient_handlers = [dict(type='ZeROGradientHandler')]
            self._logger.info(
                "Training with zero is detected, ZeROGradientHandler is automatically "
                "added even though not specified in the configuration",
                ranks=[0])
        elif gpc.is_initialized(ParallelMode.DATA) and gpc.get_world_size(
                ParallelMode.DATA) > 1:
            gradient_handlers = [dict(type='DataParallelGradientHandler')]
            self._logger.info(
                "Data parallel training is detected, DataParallelGradientHandler is automatically "
                "added even though not specified in the configuration",
                ranks=[0])

        if gradient_handlers is None:
            self._logger.warning(
                "No gradient handler is set up, please make sure you do not need "
                "to all-reduce the gradients after a training step.",
                ranks=[0])
        else:
            for cfg in gradient_handlers:
                handler = build_gradient_handler(cfg, model, optimizer)
                self._gradient_handlers.append(handler)

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def criterion(self):
        return self._criterion

    @property
    def schedule(self):
        return self._schedule

    @property
    def gradient_accumulation(self):
        return self._grad_accum_size

    def handle_gradient(self):
        """Handles all-reduce operations of gradients across different parallel groups.
        """
        for handler in self._gradient_handlers:
            handler.handle_gradient()

    def train(self):
        """Sets the model to training mode.
        """
        self.training = True
        self._model.train()

    def eval(self):
        """Sets the model to evaluation mode.
        """
        self.training = False
        self._model.eval()

    def step(self,
             data_iter,
             is_last_iteration: bool = False,
             return_loss=True):
        """A running step based on the schedule. Usually, it runs a training or
        evaluation over a batch of dataset.

        :param data_iter: Data iterator of the dataset
        :param is_last_iteration: If True, this iteration is the last iteration in the epoch
        :param return_loss: loss will be returned if True
        :type data_iter: Iterator
        :type is_last_iteration: bool, optional
        :type return_loss: bool, optional
        :return: (output, lablel, loss)
        """
        if self.training:
            self._optimizer.zero_grad()

        # differentiate training and eval with grad accum
        if self.training:
            for i in range(self._grad_accum_size):
                output, label, loss = self._schedule.forward_backward_step(
                    data_iter, self._model, self._criterion, self._optimizer,
                    forward_only=False,
                    grad_accum_size=self._grad_accum_size,
                    return_loss=return_loss)

                if i == self._grad_accum_size - 1:
                    # all reduce gradients
                    self.handle_gradient()
                    self._schedule.optimizer_step(self._model, self._optimizer, self._grad_clip)
        else:
            output, label, loss = self._schedule.forward_backward_step(
                data_iter, self._model, self._criterion, self._optimizer,
                forward_only=True,
                grad_accum_size=1,
                return_loss=return_loss)

        # consume the remaining dataset left out due to gradient accumulation
        if is_last_iteration:
            while True:
                try:
                    _ = next(data_iter)
                except StopIteration:
                    break

        return output, label, loss
