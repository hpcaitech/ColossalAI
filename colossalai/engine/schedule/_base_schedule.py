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

    @staticmethod
    def _move_tensor(element):
        if torch.is_tensor(element):
            if not element.is_cuda:
                return element.to(get_current_device()).detach()
        return element

    def _move_to_device(self, data):
        if isinstance(data, (tuple, list)):
            data = tuple([self._move_tensor(d) for d in data])
        elif torch.is_tensor(data):
            data = data.to(get_current_device()).detach()
        return data

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

    def initialize(self, model, optimizer):
        """Initializes the model and the optimizer before training.
         This is often used in FP16 training.

        :param model: The neural network model
        :param optimizer: Optimizer for updating the parameters
        """
        return model, optimizer

    @abstractmethod
    def forward_backward_step(self,
                              data_iter,
                              model,
                              criterion,
                              optimizer=None,
                              forward_only=False,
                              grad_accum_size: int = 1,
                              return_loss=True):
        """The process function over a batch of dataset for training or evaluation.

        :param data_iter: Data iterator of the dataset
        :param model: Model used in training or evaluation
        :param optimizer: Optimizer used in training or evaluation
        :param criterion: Loss function
        :param forward_only: If True, the process won't include backward
        :param grad_accum_size: Steps of gradient accumulation
        :param return_loss: If False, the loss won't be returned
        """
        pass

    @abstractmethod
    def optimizer_step(self, model, optimizer, grad_clipping: float = 0.0):
        """Updates the parameters with the optimizer.

        :param model: The neural network model
        :param optimizer: Optimizer for updating the parameters
        :param grad_clipping: The norm of gradient clipping
        :type grad_clipping: float, optional
        """
        pass
