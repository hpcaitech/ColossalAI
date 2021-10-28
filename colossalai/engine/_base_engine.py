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
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .schedule import BaseSchedule, NoPipelineSchedule


class Engine:
    """Basic engine class for training and evaluation. It runs a specific process method 
    :meth:`step` which is based on the given :attr:`schedule` over each batch of a dataset.

    :param train_dataloader: Dataloader in training
    :param test_dataloader: Dataloader in evaluation
    :param model: The neural network model
    :param criterion: Criterion for calculating loss
    :param optimizer: Optimizer for updating the parameters
    :param lr_scheduler: Learning rate scheduler ajusting learning rate during the training or evaluation
    :param schedule: Running schedule in :meth:`step`
    :type train_dataloader: DataLoader, optional
    :type test_dataloader: DataLoader, optional
    :type model: Module
    :type criterion: _Loss, optional
    :type optimizer: Optimizer, optional
    :type lr_scheduler: _LRScheduler, optional
    :type schedule: BaseSchedule, optional
    """
    def __init__(self,
                 train_dataloader: Optional[DataLoader] = None,
                 test_dataloader: Optional[DataLoader] = None,
                 model: Module = None,
                 criterion: _Loss = None,
                 optimizer: Optimizer = None,
                 lr_scheduler: Optional[_LRScheduler] = None,
                 schedule: BaseSchedule = None):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        assert model is not None, "Engine requires a model"
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.schedule = schedule if schedule is not None \
            else NoPipelineSchedule()
        self._logger = get_global_dist_logger()

        # build gradient handler
        self._gradient_handlers = []
        gradient_handler_cfg = []

        if hasattr(gpc.config, 'gradient_handler'):
            assert isinstance(gpc.config.gradient_handler, list), \
                f'argument gradient_handler_cfg expected type list, ' \
                f'but got type {type(gpc.config.gradient_handler)}'
            gradient_handler_cfg = gpc.config.gradient_handler
        elif isinstance(self.optimizer, (ZeroRedundancyOptimizer_Level_2,
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
            handler = build_gradient_handler(cfg, self.model, self.optimizer)
            self._gradient_handlers.append(handler)

        self.schedule.initialize(self.train_dataloader, self.model,
                                 self.criterion, self.optimizer,
                                 self.lr_scheduler)
        self.forward_only = False

    def handle_gradient(self):
        """Handles all-reduce operations of gradients across different parallel groups.
        """
        for handler in self._gradient_handlers:
            handler.handle_gradient()

    def set_dataloader(self, data: DataLoader, train: bool = True):
        """Sets dataloader in training or evaluation.

        :param data: Dataloader to be set
        :param train: Set training dataloader if True, otherwise evaluation dataloader
        :type data: DataLoader
        :type train: bool
        """
        if train:
            self.train_dataloader = data
        else:
            self.test_dataloader = data

    def get_model(self):
        """Returns the neural network model in the engine.
        """
        return self.model
    def get_optimizer(self):
        """Returns optimizier in the engine.
        """
        return self.optimizer

    def get_lr_scheduler(self):
        """Returns the learning rate scheduler in the engine.
        """
        return self.lr_scheduler

    def train(self):
        """Sets the model to training mode.
        """
        self.forward_only = False
        self.schedule.train(dataloader=self.train_dataloader, mode=True)

    def eval(self):
        """Sets the model to evaluation mode.
        """
        self.forward_only = True
        self.schedule.train(dataloader=self.test_dataloader, mode=False)

    def is_train(self):
        """Returns True if it is in training, otherwise False.
        """
        return not self.forward_only

    def get_lr(self):
        """Gets current learning rate.
        """
        return self.schedule.get_lr()

    def step(self, return_loss=True):
        """A running step based on the schedule. Usually, it runs a training or
        evaluation over a batch of dataset.

        :param return_loss: loss will be returned if True
        :type return_loss: bool
        :return: (output, lablel, loss)
        """
        self.schedule.zero_grad(forward_only=self.forward_only)

        output, label, loss = self.schedule.forward_backward_step(
            forward_only=self.forward_only, return_loss=return_loss)

        if not self.forward_only:
            # all reduce gradients
            self.handle_gradient()

            self.schedule.step()

        return output, label, loss
