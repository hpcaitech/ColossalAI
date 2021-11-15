#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Union

import torch.cuda
import torch.distributed as dist
from torch import Tensor

from colossalai.communication import *
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn import (ZeroRedundancyOptimizer_Level_2,
                           ZeroRedundancyOptimizer_Level_3)
from colossalai.utils import get_current_device
from ._base_schedule import BaseSchedule
from ._utils import convert_to_fp16
from ..amp import AMP_TYPE


def squeeze(x: Union[Tensor, tuple, list]):
    if isinstance(x, (tuple, list)):
        return x[0]
    else:
        return x


class PipelineSchedule(BaseSchedule):
    """A helper schedule class for pipeline parallelism running environment.
    It uses non-interleaved 1F1B strategy. Other properties are similar as
    :class:`NoPipelineSchedule`.

    :param num_microbatches: The number of microbatches
    :param amp_type: The type of automatic mixed precision
    :param amp_config: The configuration of automatic mixed procision
    :type num_microbatches: int
    :type amp_type: AMP_TYPE
    :type amp_config: dict
    """

    def __init__(self,
                 num_microbatches,
                 amp_type: AMP_TYPE = None,
                 amp_config: dict = None):
        super().__init__()

        self.num_microbatches = num_microbatches
        self.data_sync = True  # close after making sure data is identical

        # amp
        # LSGL: amp_config is not used, but leave here for future extension
        self.amp_type = amp_type
        self.amp_config = amp_config

        if self.amp_type is not None:
            assert self.amp_type == AMP_TYPE.PARALLEL, 'We only support AMP_TYPE.PARALLEL for pipeline training for now'

    def _move_to_device(self, data):
        if isinstance(data, (
                tuple,
                list,
        )):
            assert len(data) == 1, "Data tuple's length in pipeline should be 1"
            data = data[0]
        assert torch.is_tensor(data), "Data in pipeline should be tensor"
        data = data.to(get_current_device()).detach()
        return data

    def _sync_data(self):
        if gpc.is_first_rank(ParallelMode.PIPELINE):
            src_rank = gpc.get_global_rank()
            dist.broadcast(
                tensor=self.batch_data,
                src=src_rank,
                group=gpc.get_group(ParallelMode.PIPELINE_PREV)
            )
            dist.broadcast(
                tensor=self.batch_label,
                src=src_rank,
                group=gpc.get_group(ParallelMode.PIPELINE_PREV)
            )
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            src_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)
            dist.broadcast(
                tensor=self.batch_data,
                src=src_rank,
                group=gpc.get_group(ParallelMode.PIPELINE_NEXT)
            )
            dist.broadcast(
                tensor=self.batch_label,
                src=src_rank,
                group=gpc.get_group(ParallelMode.PIPELINE_NEXT)
            )

    # Pipeline schedule just puts data in memory
    def load_batch(self, data_iter):
        if data_iter is None:
            raise RuntimeError('Dataloader is not defined.')
        self.batch_pos = 0
        data, label = next(data_iter)
        self.batch_data, self.batch_label = \
            self._move_to_device(data), self._move_to_device(label)
        batch_size = self.batch_data.shape[0]
        assert batch_size % self.num_microbatches == 0, \
            "Batch size should divided by the number of microbatches"
        self.microbatch_size = batch_size // self.num_microbatches
        if self.data_sync:
            self._sync_data()

    def _get_data_slice(self, tensor):
        return tensor[self.batch_pos: self.batch_pos + self.microbatch_size]

    def load_micro_batch(self):
        data = self._get_data_slice(self.batch_data)
        label = self._get_data_slice(self.batch_label)
        self.batch_pos += self.microbatch_size
        return (data,), (label,)

    def initialize(self, model, optimizer):
        if isinstance(optimizer, (ZeroRedundancyOptimizer_Level_2, ZeroRedundancyOptimizer_Level_3)):
            raise TypeError(
                "Pipeline schedule is currently not compatible with ZeRO Level 2 and Level 3"
            )

        # LSG: set default dtype to fp16 for communication
        if self.amp_type == AMP_TYPE.PARALLEL:
            torch.set_default_dtype(torch.half)
            self.logger.info(
                'default tensor dtype is set to torch.half for fp16 training',
                ranks=[0])

    def forward_step(self, model, criterion, input_tensor, return_tensors,
                     grad_accum_size, return_loss=True):
        """Forward step for passed-in model. If it is the first stage, the input tensor 
        is obtained from data_iterator, otherwise the passed-in input_tensor is used.
        Returns output tensor. This is a helper function and can be ignored by users.
        """

        if input_tensor is None:
            input_tensor, label = self.load_micro_batch()
            if self.amp_type == AMP_TYPE.PARALLEL:
                input_tensor = convert_to_fp16(input_tensor)
        input_tensor = squeeze(input_tensor)
        output_tensor = model(input_tensor)
        output_tensor = squeeze(output_tensor)

        if gpc.is_last_rank(ParallelMode.PIPELINE):
            if return_loss:
                input_tensor, label = self.load_micro_batch()
                loss_reduced = criterion(output_tensor, *label) \
                               / (self.num_microbatches * grad_accum_size)
                return_tensors.append(
                    tuple((output_tensor, label[0], loss_reduced)))
                return loss_reduced
            else:
                return_tensors.append(output_tensor)
                return output_tensor

        else:
            return output_tensor

    def backward_step(self, optimizer, input_tensor, output_tensor, output_tensor_grad):
        """Backward step through the passed-in output tensor. If it is the last stage, the 
        output_tensor_grad is None, otherwise it is the gradients with respect to stage's output tensor.
        Returns the gradients with respect to the input tensor (None if first stage).
        This is a helper function and can be ignored by users.
        """

        # Retain the grad on the input_tensor.
        if input_tensor is not None:
            input_tensor.retain_grad()

        # Backward pass.
        if output_tensor_grad is None and self.amp_type == AMP_TYPE.PARALLEL:
            output_tensor = optimizer.scale_loss(output_tensor)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

        # Collect the grad of the input_tensor.
        input_tensor_grad = None
        if input_tensor is not None:
            input_tensor_grad = input_tensor.grad

        return input_tensor_grad

    def forward_backward_step(self,
                              data_iter,
                              model,
                              criterion,
                              optimizer=None,
                              forward_only=False,
                              grad_accum_size: int = 1,
                              return_loss=True):
        """Runs non-interleaved 1F1B schedule, with communication between pipeline stages.
        Returns a tuple with losses if the last stage, an empty tuple otherwise.

        :return: (output, label, loss)
        """

        assert forward_only or return_loss, \
            'The argument \'return_loss\' has to be True when \'forward_only\' is False, but got False.'

        self.load_batch(data_iter)
        num_warmup_microbatches = \
            (gpc.get_world_size(ParallelMode.PIPELINE) -
             gpc.get_local_rank(ParallelMode.PIPELINE) - 1)
        num_warmup_microbatches = min(num_warmup_microbatches,
                                      self.num_microbatches)
        num_microbatches_remaining = self.num_microbatches - num_warmup_microbatches

        # Input, output tensors only need to be saved when doing backward passes
        input_tensors = None
        output_tensors = None
        if not forward_only:
            input_tensors = []
            output_tensors = []
        return_tensors = []

        # Used for tensor meta information communication
        ft_shape = None
        bt_shape = None
        fs_checker = True

        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                ft_shape = recv_tensor_meta(ft_shape)
            input_tensor = recv_forward(ft_shape)
            output_tensor = self.forward_step(
                model, criterion,
                input_tensor, return_tensors,
                grad_accum_size, return_loss=return_loss
            )
            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                bt_shape = output_tensor.shape
                fs_checker = send_tensor_meta(output_tensor, fs_checker)
            send_forward(output_tensor)

            if not forward_only:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                ft_shape = recv_tensor_meta(ft_shape)
            input_tensor = recv_forward(ft_shape)

        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            last_iteration = (i == (num_microbatches_remaining - 1))

            output_tensor = self.forward_step(
                model, criterion,
                input_tensor, return_tensors,
                grad_accum_size, return_loss=return_loss
            )
            if forward_only:
                send_forward(output_tensor)

                if not last_iteration:
                    input_tensor = recv_forward(ft_shape)

            else:
                output_tensor_grad = send_forward_recv_backward(
                    output_tensor, bt_shape)

                # Add input_tensor and output_tensor to end of list.
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

                # Pop input_tensor and output_tensor from the start of the list for
                # the backward pass.
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                input_tensor_grad = self.backward_step(
                    optimizer,
                    input_tensor, output_tensor,
                    output_tensor_grad
                )

                if last_iteration:
                    input_tensor = None
                    send_backward(input_tensor_grad)
                else:
                    input_tensor = send_backward_recv_forward(
                        input_tensor_grad, ft_shape)

        # Run cooldown backward passes.
        if not forward_only:
            for i in range(num_warmup_microbatches):
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                output_tensor_grad = recv_backward(bt_shape)

                input_tensor_grad = self.backward_step(
                    optimizer,
                    input_tensor, output_tensor,
                    output_tensor_grad
                )

                send_backward(input_tensor_grad)

        if len(return_tensors) > 0:
            if return_loss:
                output, label, loss = tuple(map(list, zip(*return_tensors)))
                return (torch.cat(output, dim=0),
                        torch.cat(label, dim=0),
                        sum(loss) * grad_accum_size)
            else:
                return tuple((torch.cat(return_tensors, dim=0), None, None))
        else:
            return tuple((None, None, None))

    def optimizer_step(self, model, optimizer, grad_clipping: float = 0.0):
        optimizer.step()
