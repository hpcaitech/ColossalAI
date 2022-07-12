#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod

import torch

from typing import Iterable, Callable
from colossalai.logging import get_dist_logger
from colossalai.utils import get_current_device


class BaseSchedule(ABC):
    """A basic helper class to control the process of training or evaluation.
    It mainly composes of forward_backward_step for gradient backward and
    optimizer_step for parameters update.
    For the convenience to enable FP16, we aggregate all codes that contain the
    control of FP16 in class schedule.

    Args:
        data_process_func (Callable, optional): The preprocessing function which receives a batch of data and arranges them into data and label.
    """

    def __init__(self, data_process_func: Callable = None):
        self.logger = get_dist_logger()
        self.data_process_func = data_process_func

    @staticmethod
    def _move_tensor(element):
        if torch.is_tensor(element):
            if not element.is_cuda:
                return element.to(get_current_device()).detach()
        return element

    def _move_to_device(self, data):
        if isinstance(data, torch.Tensor):
            data = data.to(get_current_device())
        elif isinstance(data, (list, tuple)):
            data_to_return = []
            for element in data:
                if isinstance(element, dict):
                    data_to_return.append({k: self._move_tensor(v) for k, v in element.items()})
                else:
                    data_to_return.append(self._move_tensor(element))
            data = data_to_return
        elif isinstance(data, dict):
            data = {k: self._move_tensor(v) for k, v in data.items()}
        else:
            raise TypeError(
                f"Expected batch data to be of type torch.Tensor, list, tuple, or dict, but got {type(data)}")
        return data

    def _get_batch_size(self, data):
        if isinstance(data, torch.Tensor):
            return data.size(0)
        elif isinstance(data, (list, tuple)):
            if isinstance(data[0], dict):
                return data[0][list(data[0].keys())[0]].size(0)
            return data[0].size(0)
        elif isinstance(data, dict):
            return data[list(data.keys())[0]].size(0)

    def load_batch(self, data_iter, to_gpu=True):
        """Loads a batch from data iterator. It returns the data and labels which are
        already in the same GPU as where the model's.

        Args:
            data_iter (Iterable): Data iterator from which get a batch of data, obtained by calling iter(dataloader).
            to_gpu (bool, optional): Whether the data should be moved to GPU

        Returns:
            Tuple (:class:`Tensor`, :class:`torch.Tensor`): A tuple of (data, label).
        """
        if data_iter is None:
            raise RuntimeError('Dataloader is not defined.')
        batch_data = next(data_iter)

        if to_gpu:
            batch_data = self._move_to_device(batch_data)
        self.batch_size = self._get_batch_size(batch_data)
        return batch_data

    def pre_processing(self, engine):
        """To perform actions before running the schedule.
        """
        pass

    @abstractmethod
    def forward_backward_step(self,
                              engine,
                              data_iter: Iterable,
                              forward_only: bool,
                              return_loss: bool = True,
                              return_output_label: bool = True):
        """The process function over a batch of dataset for training or evaluation.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            data_iter (Iterable): Data iterator from which get a batch of data, obtained by calling iter(dataloader).
            forward_only (bool): If True, the process won't include backward.
            return_loss (bool, optional): If False, the loss won't be returned.
            return_output_label (bool, optional): If False, the output and label won't be returned.
        """
        pass

    @staticmethod
    def _call_engine(engine, inputs):
        if isinstance(inputs, torch.Tensor):
            return engine(inputs)
        elif isinstance(inputs, (list, tuple)):
            return engine(*inputs)
        elif isinstance(inputs, dict):
            return engine(**inputs)
        else:
            TypeError(
                f"Expected engine inputs to be of type torch.Tensor, list, tuple, or dict, but got {type(inputs)}")

    @staticmethod
    def _call_engine_criterion(engine, outputs, labels):
        assert isinstance(outputs,
                          (torch.Tensor, list, tuple,
                           dict)), f'Expect output of model is (torch.Tensor, list, tuple), got {type(outputs)}'
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        if isinstance(labels, torch.Tensor):
            labels = (labels,)

        if isinstance(outputs, (tuple, list)) and isinstance(labels, (tuple, list)):
            return engine.criterion(*outputs, *labels)
        elif isinstance(outputs, (tuple, list)) and isinstance(labels, dict):
            return engine.criterion(*outputs, **labels)
        elif isinstance(outputs, dict) and isinstance(labels, dict):
            return engine.criterion(**outputs, **labels)
        elif isinstance(outputs, dict) and isinstance(labels, (list, tuple)):
            raise ValueError(f"Expected labels to be a dict when the model outputs are dict, but got {type(labels)}")
        else:
            raise TypeError(f"Expected model outputs and labels to be of type torch.Tensor ' \
                '(which is auto-converted to tuple), list, tuple, or dict, ' \
                'but got {type(outputs)} (model outputs) and {type(labels)} (labels)")
