#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional
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
        batch_data_process_func (Callable, optional): The preprocessing function which receives a batch of data,
        and it will be executed in load_batch.
    """

    def __init__(self, batch_data_process_func: Callable = None):
        self.logger = get_dist_logger()
        self.batch_data_process_func = batch_data_process_func
        device = get_current_device()
        self._memcpy_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.Stream() if device.type == "cuda" else None
        )
        self._cur_batch = None
        self._connected = False

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
            data = [self._move_tensor(v) for v in data]
        elif isinstance(data, dict):
            data = {k: self._move_tensor(v) for k, v in data.items()}
        else:
            raise TypeError(
                f"Expected batch data to be of type torch.Tensor, list, tuple, or dict, but got {type(data)}")
        return data

    def _connect(self, dataloader_iter: Iterable, to_gpu: bool):
        batch_data = next(dataloader_iter)

        with torch.cuda.stream(self._memcpy_stream):
            self._cur_batch = self._move_to_device(batch_data) if to_gpu else batch_data

        self._connected = True

    def _wait_for_batch(self, batch, stream: Optional[torch.cuda.streams.Stream]):
        if stream is None:
            return
        
        torch.cuda.current_stream().wait_stream(stream)
        cur_stream = torch.cuda.current_stream()
        
        if isinstance(batch, torch.Tensor):
            batch = batch.record_stream(cur_stream)
        elif isinstance(batch, (list, tuple)):
            for v in batch:
                v.record_stream(cur_stream)
        elif isinstance(batch, dict):
            for _, v in batch.items():
                v.record_stream(cur_stream)
        else:
            raise TypeError(
                f"Expected batch data to be of type torch.Tensor, list, tuple, or dict, but got {type(data)}")
        # batch.record_stream(cur_stream)

    def _get_batch_size(self, data):
        if isinstance(data, torch.Tensor):
            return data.size(0)
        elif isinstance(data, (list, tuple)):
            return data[0].size(0)
        elif isinstance(data, dict):
            return data[next(data.keys())].size(0)
    
    # def load_batch(self, data_iter, to_gpu=True):
    #     if data_iter is None:
    #         raise RuntimeError('Dataloader is not defined.')
    #     batch_data = next(data_iter)

    #     if to_gpu:
    #         batch_data = self._move_to_device(batch_data)
    #     self.batch_size = self._get_batch_size(batch_data)
    #     return batch_data

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
        
        if not self._connected:
            self._connect(data_iter, to_gpu)

        try:
            next_batch = next(data_iter)
        except StopIteration:
            next_batch = self._cur_batch

        self.batch_size = self._get_batch_size(next_batch)
        
        cur_batch = self._cur_batch
        self._cur_batch = next_batch

        return cur_batch

    def preload_batch(self, to_gpu: bool = True) -> None:

        with torch.cuda.stream(self._memcpy_stream):
            self._cur_batch = self._move_to_device(self._cur_batch) if to_gpu else self._cur_batch
            self._label = self._move_to_device(self._cur_batch) if to_gpu else self._cur_batch
        
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
        else:
            return engine(**inputs)

    @staticmethod
    def _call_engine_criterion(engine, outputs, labels):
        assert isinstance(
            outputs,
            (torch.Tensor, list, tuple)), f'Expect output of model is (torch.Tensor, list, tuple), got {type(outputs)}'
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        if isinstance(labels, torch.Tensor):
            return engine.criterion(*outputs, labels)
        else:
            return engine.criterion(*outputs, **labels)
