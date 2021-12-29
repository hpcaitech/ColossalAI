#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod

import torch

from torch import Tensor
from typing import Iterable, Union, List, Callable
from .._base_engine import Engine
from colossalai.logging import get_dist_logger
from colossalai.utils import get_current_device
from colossalai.nn.layer import split_batch

class BaseSchedule(ABC):
    """A basic helper class to control the process of training or evaluation.
    It mainly composes of forward_backward_step for gradient backward and
    optimizer_step for parameters update.
    For the convenience to enable FP16, we aggreate all codes that contain the
    control of FP16 in class schedule.
    """

    def __init__(self, batch_data_process_func: Callable = None):
        self.logger = get_dist_logger()
        self.batch_data_process_func = batch_data_process_func

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

    def _to_list(self, data):
        if torch.is_tensor(data):
            return [data]
        return data

    def load_batch(self, data_iter):
        """Loads a batch from data iterator. It returns the data and labels which are
        already in the same GPU as where the model's.

        :return: (data, label)
        :rtype: (:class:`Tensor`, :class:`torch.Tensor`)
        """
        if data_iter is None:
            raise RuntimeError('Dataloader is not defined.')
        batch_data = next(data_iter)

        if self.batch_data_process_func:
            data, label = self.batch_data_process_func(batch_data)
        else:
            data, label = batch_data

        if isinstance(label, (tuple, list)):
            self.batch_size = label[0].size(0)
        else:
            self.batch_size = label.size(0)
        data, label = self._to_list(split_batch(data)), self._to_list(split_batch(label))
        return self._move_to_device(data), self._move_to_device(label)

    def pre_processing(self, engine: Engine):
        """To perform actions before running the schedule.
        """
        pass

    @abstractmethod
    def forward_backward_step(self,
                              engine: Engine,
                              data_iter: Iterable,
                              forward_only: bool,
                              return_loss: bool = True
                              ):
        """The process function over a batch of dataset for training or evaluation.

        :param engine: Colossalai training engine
        :param inputs: input data
        :param labels: ground truth
        :param forward_only: If True, the process won't include backward
        :param return_loss: If False, the loss won't be returned
        """
        pass