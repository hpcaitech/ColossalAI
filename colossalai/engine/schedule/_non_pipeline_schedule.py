#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Iterable

import torch

from ._base_schedule import BaseSchedule
from colossalai.utils import conditional_context


class NonPipelineSchedule(BaseSchedule):
    """A helper schedule class for no pipeline parallelism running environment.
    During one process, it loads a batch of dataset and feeds it to the model.
    After getting the output and calculating the loss, it will use :meth:`step`
    to update the parameters if it is in training mode.

    Args:
        batch_data_process_func (Callable, optional): The preprocessing function which receives a batch of data,
        and it will be executed in load_batch.
    """

    def forward_backward_step(self,
                              engine,
                              data_iter: Iterable,
                              forward_only: bool = False,
                              return_loss: bool = True,
                              return_output_label: bool = True):
        """The process function that loads a batch of dataset and feeds it to the model.
        The returned labels and loss will None if :attr:`return_loss` is False.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                If True, the model is run for the forward pass, else back propagation will be executed.
            return_loss (bool, optional): Loss will be returned if True.
            return_output_label (bool, optional): Output and label will be returned if True.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
        """
        assert forward_only or return_loss, \
            "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."
        data, label = self.load_batch(data_iter)

        # forward
        with conditional_context(torch.no_grad(), enable=forward_only):
            output = self._call_engine(engine, data)
            if return_loss:
                loss = self._call_engine_criterion(engine, output, label)

        if not forward_only:
            engine.backward(loss)

        if return_output_label:
            if return_loss:
                return output, label, loss
            else:
                return output, label, None
        else:
            if return_loss:
                return None, None, loss
            else:
                return None, None, None
