#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from typing import Any
from torch.optim import Optimizer
from torch.distributed import ReduceOp
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.nn.optimizer import ColossalaiOptimizer
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from ._fp16_optimizer import FP16Optimizer


class NaiveAMPOptimizer(ColossalaiOptimizer):
    """A wrapper class for optimizer to cast all parameters to fp16

    :param optim: A normal optimizer like Adam or SGD
    :param args: Args used to initialize FP16 optimizer
    :param kwargs: Kwargs used to initialize FP16 optimizer

    :type optim: torch.optim.Optimizer
    """

    def __init__(self, optim: Optimizer, *args, **kwargs):
        optim = FP16Optimizer(optimizer=optim, *args, **kwargs)
        super().__init__(optim)

    def backward(self, loss: Tensor):
        """Backward with gradient scaler

        :param loss: loss computed by a loss function
        :type loss: torch.Tensor
        """
        loss = self.optim.scale_loss(loss)
        loss.backward()

    def step(self):
        return self.optim.step()

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        pass


class NaiveAMPModel(nn.Module):
    """A wrapper class for model to cast the model into fp16 and
    automatically cast the input and output
    """

    def __init__(self,
                 model: nn.Module,
                 output_to_fp32: bool = True,
                 parallel_mode: ParallelMode = ParallelMode.DATA,
                 sync_buffer: bool = True):
        super().__init__()
        self.model = model.half()
        self._output_to_fp32 = output_to_fp32
        self._sync_buf = sync_buffer

        if gpc.is_initialized(parallel_mode) and gpc.get_world_size(parallel_mode) > 1:
            self._process_group = gpc.get_group(parallel_mode)
            self._world_size = gpc.get_world_size(parallel_mode)
        else:
            self._process_group = None
            self._world_size = 1
            self._sync_buf = False
        self._first_eval_run = False

    @property
    def sync_buffer(self):
        return self._sync_buf

    @sync_buffer.setter
    def sync_buffer(self, state: bool):
        self._sync_buf = state

    def _convert_to_fp16(self, input_: Any):
        if isinstance(input_, Tensor) and input_.dtype == torch.float32:
            input_ = input_.half()
        return input_

    def _convert_to_fp32(self, input_: Any):
        if isinstance(input_, Tensor) and input_.dtype == torch.float16:
            input_ = input_.float()
        return input_

    def _reduce_module_buffer(self):
        """
        All-reduce the buffers (e.g. running stats of batch normalization) across
        data parallel ranks so that all the ranks will produce consistent results
        when given the same input
        """
        buf_list = []

        # find valid buffers
        for buf in self.model.buffers():
            if buf is not None:
                buf_list.append(buf)

        # reduce buffers across data parallel ranks
        if buf_list:
            coalesced_buf = _flatten_dense_tensors(buf_list)
            coalesced_buf.div_(self._world_size)
            dist.all_reduce(coalesced_buf, op=ReduceOp.SUM, group=self._process_group)
            unflattened_buf_list = _unflatten_dense_tensors(coalesced_buf, buf_list)
            for old, new in zip(buf_list, unflattened_buf_list):
                old.copy_(new)

    def eval(self):
        self.model.eval()

        # we only sync buffer in the first eval iteration
        # so that future eval iterations can be done without communication
        self._first_eval_run = True

    def forward(self, *args, **kwargs):
        # reduce buffers after forward will lead to error
        # as we cannot change the variables needed for gradient computation after forward
        # so we sync buffer before forward
        if (self.training or self._first_eval_run) and self._sync_buf:
            with torch.no_grad():
                self._reduce_module_buffer()

            if self._first_eval_run:
                self._first_eval_run = False

        if args:
            args = [self._convert_to_fp16(arg) for arg in args]
        if kwargs:
            for k, v in kwargs.items():
                kwargs[k] = self._convert_to_fp16(v)

        out = self.model(*args, **kwargs)

        if self._output_to_fp32:
            if isinstance(out, Tensor):
                out = self._convert_to_fp32(out)
            elif isinstance(out, (tuple, list)):
                out = [self._convert_to_fp32(val) for val in out]
        return out
