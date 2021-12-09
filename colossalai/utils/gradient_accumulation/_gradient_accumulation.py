#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn
from torch import Tensor
from typing import Iterable, Any
from colossalai.nn.optimizer import ColossalaiOptimizer
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from colossalai.utils import conditional_context
from colossalai.engine import BaseGradientHandler


class GradAccumOptimizer(ColossalaiOptimizer):

    def __init__(self, optim: Optimizer, accumulate_size: int, model: nn.Module = None):
        super().__init__(optim)
        self.accumulate_size = accumulate_size
        self.accumulate_step = 0

        # handle pytorch ddp auto all reduce
        self.model = model
        self.is_torch_ddp = isinstance(self.model, DistributedDataParallel)

    def zero_grad(self, *args, **kwargs):
        if self.accumulate_step == 0:
            self.optim.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        if self.accumulate_step < self.accumulate_size:
            return None
        else:
            self.accumulate_step = 0
            return self.optim.step(*args, **kwargs)

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        if self.accumulate_step < self.accumulate_size:
            pass
        else:
            self.optim.clip_grad_norm(model, max_norm)

    def backward(self, loss: Tensor):
        self.accumulate_step += 1

        if self.is_torch_ddp:
            no_sync = self.accumulate_step < self.accumulate_size
            with conditional_context(self.model.no_sync(), enable=no_sync):
                scaled_loss = loss / self.accumulate_size
                self.optim.backward(scaled_loss)
        else:
            scaled_loss = loss / self.accumulate_size
            self.optim.backward(scaled_loss)

    def backward_by_grad(self, tensor: Tensor, grad: Tensor):
        no_sync = self.is_torch_ddp and self.accumulate_step < self.accumulate_size

        if no_sync:
            with self.model.no_sync():
                self.optim.backward_by_grad(tensor, grad)
        else:
            self.optim.backward_by_grad(tensor, grad)


class GradAccumDataloader():

    def __init__(self, dataloader: Iterable, accumulate_size: int) -> None:
        self.dataloader = dataloader
        self.consume_remain_data = not isinstance(dataloader, DataLoader)
        self.steps_per_epoch = len(dataloader) - len(dataloader) % accumulate_size

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.dataloader, __name)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        self._cur_step = 0
        self._dataiter = iter(self.dataloader)
        return self

    def __next__(self) -> Any:
        if self._cur_step < self.steps_per_epoch:
            self._cur_step += 1

            if self._cur_step == self.steps_per_epoch and self.consume_remain_data:
                # this is to handle non standard pytorch dataloader
                # such as dali dataloader
                while True:
                    try:
                        _ = next(self._dataiter)
                    except StopIteration:
                        break
            return next(self._dataiter)
        else:
            raise StopIteration


class GradAccumLrSchedulerByStep(_LRScheduler):

    def __init__(self, lr_scheduler: _LRScheduler, accumulate_size: int) -> None:
        self.lr_scheduler = lr_scheduler
        self.accumulate_size = accumulate_size
        self.accumulate_step = 0

    @staticmethod
    def compute_effective_steps_per_epoch(dataloader: Iterable, accumulate_size: int):
        return len(dataloader) // accumulate_size

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.lr_scheduler, __name)

    def step(self, *args, **kwargs):
        self.accumulate_step += 1
        if self.accumulate_step < self.accumulate_size:
            pass
        else:
            self.accumulate_step = 0
            self.lr_scheduler.step(*args, **kwargs)

    def get_lr(self):
        return self.lr_scheduler.get_lr()

    def get_last_lr(self):
        return self.lr_scheduler.get_last_lr()

    def print_lr(self, *args, **kwargs):
        self.lr_scheduler.print_lr(*args, **kwargs)

    def state_dict(self) -> dict:
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.lr_scheduler.load_state_dict(state_dict)


class GradAccumGradientHandler():

    def __init__(self, grad_handler: BaseGradientHandler, accumulate_size: int) -> None:
        assert isinstance(grad_handler, BaseGradientHandler), \
            f'expected grad_handler to be type BaseGradientHandler, but got {type(grad_handler)}'
        self.grad_handler = grad_handler
        self.accumulate_size = accumulate_size
        self.accumulate_step = 0

    def handle_gradient(self):
        self.accumulate_step += 1
        if self.accumulate_step < self.accumulate_size:
            pass
        else:
            self.accumulate_step = 0
            self.grad_handler.handle_gradient()
