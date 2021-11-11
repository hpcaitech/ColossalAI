#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional

from colossalai.builder import build_gradient_handler
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger
from colossalai.nn import (ZeroRedundancyOptimizer_Level_2,
                           ZeroRedundancyOptimizer_Level_3)
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from .schedule import BaseSchedule, NoPipelineSchedule


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
                 step_schedule: BaseSchedule = None,
                 gradient_accumulation: int = 1,
                 gradient_clipping: float = 0.0):
        self.schedule = step_schedule if step_schedule is not None \
            else NoPipelineSchedule()
        self.schedule.initialize(model, optimizer)
        self.grad_accum_size = gradient_accumulation
        self.grad_accum_cur_step = 0
        self.grad_clip = gradient_clipping
        self.training = True  # default
        self._logger = get_global_dist_logger()

        # build gradient handler
        self._gradient_handlers = []
        gradient_handler_cfg = []

        if hasattr(gpc.config, 'gradient_handler'):
            assert isinstance(gpc.config.gradient_handler, list), \
                f'argument gradient_handler_cfg expected type list, ' \
                f'but got type {type(gpc.config.gradient_handler)}'
            gradient_handler_cfg = gpc.config.gradient_handler
        elif isinstance(optimizer, (ZeroRedundancyOptimizer_Level_2,
                                    ZeroRedundancyOptimizer_Level_3)):
            gradient_handler_cfg = [dict(type='ZeROGradientHandler')]
            self._logger.info(
                "Training with zero is detected, ZeROGradientHandler is automatically "
                "added even though not specified in the configuration",
                ranks=[0])
        elif gpc.is_initialized(ParallelMode.DATA) and gpc.get_world_size(
                ParallelMode.DATA) > 1:
            gradient_handler_cfg = [dict(type='DataParallelGradientHandler')]
            self._logger.info(
                "Data parallel training is detected, DataParallelGradientHandler is automatically "
                "added even though not specified in the configuration",
                ranks=[0])
        if len(gradient_handler_cfg) == 0:
            self._logger.warning(
                "No gradient handler is set up, please make sure you do not need "
                "to all-reduce the gradients after a training step.",
                ranks=[0])
        for cfg in gradient_handler_cfg:
            handler = build_gradient_handler(cfg, model, optimizer)
            self._gradient_handlers.append(handler)

    def handle_gradient(self):
        """Handles all-reduce operations of gradients across different parallel groups.
        """
        for handler in self._gradient_handlers:
            handler.handle_gradient()

    def train(self):
        """Sets the model to training mode.
        """
        self.training = True

    def eval(self):
        """Sets the model to evaluation mode.
        """
        self.training = False

    def step(self,
             data_iter,
             model: Module,
             criterion: _Loss,
             optimizer: Optimizer = None,
             is_last_iteration: bool = False,
             return_loss=True):
        """A running step based on the schedule. Usually, it runs a training or
        evaluation over a batch of dataset.

        :param data_iter: Data iterator of the dataset
        :param model: The neural network model
        :param criterion: Loss function used to calculate
        :param optimizer: Optimizer for updating the parameters
        :param is_last_iteration: If True, this iteration is the last iteration in the epoch
        :param return_loss: loss will be returned if True
        :type data_iter: Iterator
        :type model: Module
        :type criterion: _Loss
        :type optimizer: Optimizer, optional
        :type is_last_iteration: bool, optional
        :type return_loss: bool, optional
        :return: (output, lablel, loss)
        """
        if self.training and self.grad_accum_cur_step == 0:
            optimizer.zero_grad()

        output, label, loss = self.schedule.forward_backward_step(
            data_iter, model, criterion, optimizer,
            forward_only=not self.training,
            grad_accum_size=self.grad_accum_size,
            return_loss=return_loss)

        if self.training:
            self.grad_accum_cur_step += 1
            if self.grad_accum_cur_step == self.grad_accum_size:
                # all reduce gradients
                self.handle_gradient()
                self.schedule.optimizer_step(model, optimizer, self.grad_clip)
                self.grad_accum_cur_step = 0

        if is_last_iteration:
            while True:
                try:
                    trash = next(data_iter)
                except StopIteration:
                    break

        return output, label, loss
