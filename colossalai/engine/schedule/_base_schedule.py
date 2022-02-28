#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod

import torch

from typing import Iterable, Callable
from .._base_engine import Engine
from colossalai.logging import get_dist_logger
from colossalai.utils import get_current_device


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
        if isinstance(data, dict):
            data = {k: self._move_tensor(v) for k, v in data.items()}
        else:
            data = self._move_tensor(data)
        return data

    @staticmethod
    def _check_sanity(data, tag: str):
        assert isinstance(data, (torch.Tensor, dict)), \
            f'{tag} must be torch.Tensor or dict'

    def load_batch(self, data_iter, to_gpu=True):
        """Loads a batch from data iterator. It returns the data and labels which are
        already in the same GPU as where the model's.

        :param data_iter: Data iterator from which get a batch of data
        :type data_iter: DataIter
        :param to_gpu: Whether the data should be moved to GPU
        :type to_gpu: bool, optional

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
        self._check_sanity(data, 'data')
        self._check_sanity(label, 'label')
        if isinstance(data, torch.Tensor):
            self.batch_size = data.size(0)
        else:
            self.batch_size = next(iter(data.values())).size(0)
        if to_gpu:
            return self._move_to_device(data), self._move_to_device(label)
        return data, label

    def pre_processing(self, engine: Engine):
        """To perform actions before running the schedule.
        """
        pass

    @abstractmethod
    def forward_backward_step(self,
                              engine: Engine,
                              data_iter: Iterable,
                              forward_only: bool,
                              return_loss: bool = True,
                              return_output_label: bool = True
                              ):
        """The process function over a batch of dataset for training or evaluation.

        :param engine: Colossalai training engine
        :type engine: colossalai.engine.Engine
        :param data_iter: Data iterator from which get a batch of data
        :type data_iter: DataIter
        :param forward_only: If True, the process won't include backward
        :type forward_only: bool
        :param return_loss: If False, the loss won't be returned
        :type return_loss: bool, optional
        :param return_output_label: If False, the output and label won't be returned
        :type return_output_label: bool, optional
        """
        pass

    @staticmethod
    def _call_engine(engine, inputs):
        if isinstance(inputs, torch.Tensor):
            return engine(inputs)
        else:
            return engine(**inputs)

    @staticmethod
    def _call_engine_criterion(engine, outputs, labels):
        assert isinstance(outputs, (torch.Tensor, list, tuple)
                          ), f'Expect output of model is (torch.Tensor, list, tuple), got {type(outputs)}'
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        if isinstance(labels, torch.Tensor):
            return engine.criterion(*outputs, labels)
        else:
            return engine.criterion(*outputs, **labels)
