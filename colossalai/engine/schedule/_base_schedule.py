#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod

import torch

from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger
from colossalai.utils import get_current_device


class BaseSchedule(ABC):
    """A basic helper class to control the process of training or evaluation.
    It mainly composes of forward_backward_step for gradient backward and
    optimizer_step for parameters update.
    For the convenience to enable FP16, we aggreate all codes that contain the
    control of FP16 in class schedule.
    """

    def __init__(self):
        self.logger = get_global_dist_logger()
        self.grad_accum = 1
        self.training = False

    @staticmethod
    def _move_tensor(element):
        if torch.is_tensor(element):
            if not element.is_cuda:
                return element.to(get_current_device()).detach()
        return element

    def initialize(self,
                   dataloader=None,
                   model=None,
                   criterion=None,
                   optimizer=None):
        """Initializes the schedule and set parameters before running.

        :param dataloader: DataLoader in training or evaluation
        :param model: The neural network model
        :param criterion: Criterion for calculating loss
        :param optimizer: Optimizer for updating the parameters
        """
        self.dataloader = dataloader
        assert model is not None, "Schedule requires a model"
        self.model = model
        assert criterion is not None, "Schedule requires a criterion"
        self.criterion = criterion
        assert optimizer is not None, "Schedule requires an optimizer"
        self.optimizer = optimizer
        self.initialized = True

    def check_initialized(self):
        """Checks whether the schedule is initialized.
        """
        assert self.initialized, \
            'Schedule is not initialized. Call schedule.initialize(...) before using it.'

    def load_batch(self, data_iter):
        """Loads a batch from data iterator. It returns the data and labels which are
        already in the same GPU as where the model's.

        :return: (data, label)
        :rtype: (Tensor, Tensor)
        """
        if data_iter is None:
            raise RuntimeError('Dataloader is not defined.')
        data, label = next(data_iter)
        return self._move_to_device(data), self._move_to_device(label)

    def consume_batch(self):
        while True:
            try:
                self.load_batch()
            except StopIteration:
                break

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
        self.training = mode
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
        return model, optimizer

    def step(self):
        """Updates the parameters and learning rate with the optimizer.
        """
        self.check_initialized()
        self.optimizer.step()

    @abstractmethod
    def optimizer_step(self, model, optimizer, grad_clipping: float = 0.0):
        """Updates the parameters with the optimizer.

        :param model: The neural network model
        :param optimizer: Optimizer for updating the parameters
        :param grad_clipping: The norm of gradient clipping
        :type grad_clipping: float, optional
        """
        pass
