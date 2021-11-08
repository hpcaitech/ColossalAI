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
                 train_dataloader: Optional[DataLoader] = None,
                 test_dataloader: Optional[DataLoader] = None,
                 model: Module = None,
                 criterion: _Loss = None,
                 optimizer: Optimizer = None,
                 lr_scheduler: Optional[_LRScheduler] = None,
                 schedule: BaseSchedule = None,
                 gradient_accumulation: int = 1,
                 lr_scheduler_step: str = 'epoch'):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        assert model is not None, "Engine requires a model"
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.schedule = schedule if schedule is not None \
            else NoPipelineSchedule()
        self.grad_accum_size = gradient_accumulation
        self.grad_accum_step = 0
        self.lr_step = 0  # for epoch updating
        if lr_scheduler_step != 'epoch':
            self.lr_step = 1
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

        self.schedule.initialize(self.train_dataloader, self.model,
                                 self.criterion, self.optimizer)
        self.schedule.grad_accum = self.grad_accum_size
        # add for robustness
        if self.optimizer is None:
            self.forward_only = True
        else:
            self.forward_only = False

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def criterion(self):
        return self._criterion

    @property
    def schedule(self):
        return self._schedule

    def get_model(self):
        """Returns the neural network model in the engine.
        """
        return self.model

    def get_optimizer(self):
        """Returns optimizier in the engine.
        """
        return self.optimizer

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

    def is_train(self):
        """Returns True if it is in training, otherwise False.
        """
        return not self.forward_only

    def get_lr(self):
        """Gets current learning rate.
        """
        if self.lr_scheduler is not None:
            return self.lr_scheduler.get_lr()[0]
        else:
            return self.optimizer.param_groups[0]['lr']

    def step(self, return_loss=True):
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
        if not self.forward_only and self.grad_accum_step == 0:
            self.schedule.zero_grad()

        output, label, loss = self.schedule.forward_backward_step(
            forward_only=self.forward_only, return_loss=return_loss)

        if not self.forward_only:
            self.grad_accum_step += 1
            if self.grad_accum_step == self.grad_accum_size:
                # all reduce gradients
                self.handle_gradient()
                self.schedule.step()
                if self.lr_scheduler is not None and self.lr_step:
                    self.lr_scheduler.step()
                self.grad_accum_step = 0

        return output, label, loss

    def complete(self):
        """Updating after a epoch.
        """
        self.schedule.consume_batch()
        if self.lr_scheduler is not None and self.lr_step == 0:
            self.lr_scheduler.step()
