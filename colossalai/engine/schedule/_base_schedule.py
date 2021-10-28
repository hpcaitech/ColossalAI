#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod

import torch

from colossalai.logging import get_global_dist_logger
from colossalai.utils import get_current_device


class BaseSchedule(ABC):
    """A basic helper class to control the process of training or evaluation.
    """
    def __init__(self):
        self.initialized = False
        self.logger = get_global_dist_logger()

    @property
    @abstractmethod
    def num_steps(self):
        """The number of batches in training or evaluation.
        """
        pass

    def initialize(self,
                   dataloader=None,
                   model=None,
                   criterion=None,
                   optimizer=None,
                   lr_scheduler=None):
        """Initializes the schedule and set parameters before running.

        :param dataloader: DataLoader in training or evaluation
        :param model: The neural network model
        :param criterion: Criterion for calculating loss
        :param optimizer: Optimizer for updating the parameters
        :param lr_scheduler: Learning rate scheduler in the process
        """
        self.dataloader = dataloader
        assert model is not None, "Schedule requires a model"
        self.model = model
        assert criterion is not None, "Schedule requires a criterion"
        self.criterion = criterion
        assert optimizer is not None, "Schedule requires an optimizer"
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.initialized = True

    def check_initialized(self):
        """Checks whether the schedule is initialized.
        """
        assert self.initialized, \
            'Schedule is not initialized. Call schedule.initialize(...) before using it.'

    def load_batch(self):
        """Loads a batch of dataset. It returns the data and labels which are
        already in the same GPU as where the model's.

        :return: (data, label)
        :rtype: (Tensor, Tensor) 
        """
        self.check_initialized()
        if self.data_iter is None:
            raise RuntimeError('Dataloader is not defined.')
        data, label = next(self.data_iter)
        return self._move_to_device(data), self._move_to_device(label)

    def _move_to_device(self, data):
        if isinstance(data, (
                tuple,
                list,
        )):
            data = tuple([
                d.to(get_current_device()).detach() for d in data
                if torch.is_tensor(d)
            ])
        elif torch.is_tensor(data):
            data = data.to(get_current_device()).detach()
        return data

    def train(self, dataloader=None, mode=True):
        """Sets the dataloader to be used and turn the model to 
        training or evaluation mode.

        :param dataloader: Dataloader to be used
        :param mode: If True, the model will set as training mode. Otherwise, evaluation mode.
        """
        self.check_initialized()
        if mode:
            self.model.train()
        else:
            self.model.eval()
        if dataloader is not None:
            self.dataloader = dataloader
            self.data_iter = iter(dataloader)

    def zero_grad(self, forward_only=False):
        """Cleans gradients with the optimizer.
        """
        if not forward_only:
            self.check_initialized()
            self.optimizer.zero_grad()

    def get_lr(self):
        """Returns the current learning rate.
        """
        if self.lr_scheduler is not None:
            return self.lr_scheduler.get_lr()[0]
        else:
            return self.optimizer.param_groups[0]['lr']

    def step(self):
        """Updates the parameters and learning rate with the optimizer.
        """
        self.check_initialized()
        self.optimizer.step()
        # update lr scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    @abstractmethod
    def forward_backward_step(self, forward_only=False, return_loss=True):
        """The process function over a batch of dataset for training or evaluation.

        :param forward_only: If True, the process won't include backward.
        :param return_loss: If False, the loss won't be returned.
        """
        pass
