#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed import ReduceOp
from torch.optim import Optimizer

from colossalai.interface import OptimizerWrapper
from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc

from ._fp16_optimizer import FP16Optimizer


class NaiveAMPOptimizer(OptimizerWrapper):
    """A wrapper class for optimizer to cast all parameters to fp16

    Args:
        optim (torch.optim.Optimizer): A normal optimizer like Adam or SGD.
        grad_scaler (BaseGradScaler): grad scaler for gradient chose in
                                      ``constant_grad_scaler`` or ``dynamic_grad_scaler``.
        clip_grad_norm (float, optional): clip gradients with this global L2 norm. Default 0.
        verbose (bool, optional): if set to `True`, will print debug info. Default False.

    Note:
        clipping is ignored if ``clip_grad_norm`` equals 0.
    """

    def __init__(self, optim: Optimizer, *args, **kwargs):
        optim = FP16Optimizer(optim, *args, **kwargs)
        super().__init__(optim)

    def backward(self, loss: Tensor):
        self.optim.backward(loss)

    def step(self):
        return self.optim.step()

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        if self.optim.max_norm == max_norm:
            return
        raise RuntimeError(
            "NaiveAMP optimizer has clipped gradients during optimizer.step(). "
            "If you have supplied clip_grad_norm in the amp_config, "
            "executing the method clip_grad_norm is not allowed."
        )


class NaiveAMPModel(nn.Module):
    r"""A wrapper class for model to cast the model into fp16 and
    automatically cast the input and output

    Args:
        model (torch.nn.Module): torch.nn.Module to be wrapped.
        output_to_fp32 (bool, optional): Whether cast output of this module into fp32. (Default: True)
        parallel_mode (:class:`colossalai.legacy.context.ParallelMode`): Parallel group mode used in this module.
                                                                  (Default: ``ParallelMode.DATA``)
        sync_buffer (bool, optional): whether to synchronize buffer. (Default: True)

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """

    def __init__(
        self,
        model: nn.Module,
        output_to_fp32: bool = True,
        parallel_mode: ParallelMode = ParallelMode.DATA,
        sync_buffer: bool = True,
    ):
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
            elif isinstance(out, dict):
                out = {key: self._convert_to_fp32(val) for key, val in out.items()}
        return out
