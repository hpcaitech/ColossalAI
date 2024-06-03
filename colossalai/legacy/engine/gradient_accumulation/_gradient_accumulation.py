#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Any, Iterable, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from colossalai.interface import OptimizerWrapper
from colossalai.legacy.engine import BaseGradientHandler
from colossalai.utils import conditional_context


class GradAccumOptimizer(OptimizerWrapper):
    """A wrapper for the optimizer to enable gradient accumulation by skipping the steps
    before accumulation size is reached.

    Args:
        optim (:class:`torch.optim.Optimizer`): Your optimizer object for gradient accumulation.
        accumulate_size (int): The number of steps to accumulate gradients.
        model (:class:`torch.nn.Module`):
            Your model object to check if it is DistributedDataParallel for special handling of no_sync() context.
    """

    def __init__(self, optim: Optimizer, accumulate_size: int, model: nn.Module = None):
        super().__init__(optim)
        self.accumulate_size = accumulate_size
        self.accumulate_step = 0

        # handle pytorch ddp auto all reduce
        self.model = model
        self.is_torch_ddp = isinstance(self.model, DistributedDataParallel)

    def zero_grad(self, *args, **kwargs) -> None:
        """
        Set all gradients to zero.

        Args:
            *args: positional arguments for the optimizer wrapped
            **kwargs: keyword arguments for the optimizer wrapped
        """

        if self.accumulate_step == 0:
            self.optim.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs) -> None:
        """
        Update the model parameters.

        Args:
            *args: positional arguments for the optimizer wrapped
            **kwargs: keyword arguments for the optimizer wrapped
        """

        if self.accumulate_step < self.accumulate_size:
            return None
        else:
            self.accumulate_step = 0
            return self.optim.step(*args, **kwargs)

    def clip_grad_norm(self, model: nn.Module, max_norm: float) -> None:
        """
        Clip gradients by norm.

        Args:
            model (:class:`torch.nn.Module`): a torch module instance
            max_norm (float): the max norm for gradient clipping
        """

        if self.accumulate_step < self.accumulate_size:
            pass
        else:
            self.optim.clip_grad_by_norm(max_norm)

    def backward(self, loss: Tensor) -> None:
        """Execute backward pass.

        Args:
            loss (:class:`torch.Tensor`): the loss value.
        """

        self.accumulate_step += 1

        if self.is_torch_ddp:
            no_sync = self.accumulate_step < self.accumulate_size
            with conditional_context(self.model.no_sync(), enable=no_sync):
                scaled_loss = loss / self.accumulate_size
                self.optim.backward(scaled_loss)
        else:
            scaled_loss = loss / self.accumulate_size
            self.optim.backward(scaled_loss)

    def backward_by_grad(self, tensor: Tensor, grad: Tensor) -> None:
        """Execute backward pass given the gradients of the output.

        Args:
            loss (:class:`torch.Tensor`): the loss value.
            grad (:class:`torch.Tensor`): the output gradient.
        """

        self.accumulate_step += 1
        no_sync = self.is_torch_ddp and self.accumulate_step < self.accumulate_size

        if no_sync:
            with self.model.no_sync():
                self.optim.backward_by_grad(tensor, grad)
        else:
            self.optim.backward_by_grad(tensor, grad)


class GradAccumDataloader:
    """A wrapper for dataloader to enable gradient accumulation by dropping the last incomplete steps.

    Note:
        The dataloader would drop the last incomplete steps for gradient accumulation.
        For example, if a dataloader has 10 batches of data and accumulate size is 4. The model parameters will
        be updated only twice at step 4 and step 8. The last two batches of data do not form a complete 4-step cycle.
        Thus, they will be automatically skipped by this class. If the dataloader is not standard PyTorch dataloader,
        (e.g. Dali dataloader), this class will automatically consume (load data for nothing) the remaining 2 batches.

    Args:
        dataloader (``Iterable``): Your dataloader object for gradient accumulation.
        accumulate_size (int): The number of steps to accumulate gradients.
    """

    def __init__(self, dataloader: Iterable, accumulate_size: int) -> None:
        self.dataloader = dataloader
        self.consume_remain_data = not isinstance(dataloader, DataLoader)
        self.steps_per_epoch = len(dataloader) - len(dataloader) % accumulate_size

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.dataloader, __name)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterable:
        self._cur_step = 0
        self._dataiter = iter(self.dataloader)
        return self

    def __next__(self) -> Union[Tensor, Tuple[Tensor]]:
        if self._cur_step < self.steps_per_epoch:
            self._cur_step += 1
            data = next(self._dataiter)

            if self._cur_step == self.steps_per_epoch and self.consume_remain_data:
                # this is to handle non standard pytorch dataloader
                # such as dali dataloader
                while True:
                    try:
                        _ = next(self._dataiter)
                    except StopIteration:
                        break
            return data
        else:
            raise StopIteration


class GradAccumLrSchedulerByStep(_LRScheduler):
    """A wrapper for the LR scheduler to enable gradient accumulation by skipping the steps
    before accumulation size is reached.

    Args:
        lr_scheduler (:class:`torch.optim.lr_scheduler._LRScheduler`):
            Your ``lr_scheduler`` object for gradient accumulation.
        accumulate_size (int): The number of steps to accumulate gradients.
    """

    def __init__(self, lr_scheduler: _LRScheduler, accumulate_size: int) -> None:
        self.lr_scheduler = lr_scheduler
        self.accumulate_size = accumulate_size
        self.accumulate_step = 0

    @staticmethod
    def compute_effective_steps_per_epoch(dataloader: Iterable, accumulate_size: int) -> int:
        """
        Computes the number of effective training iterations. An effective iteration is defined
        as the the aggregation of <accumulate_size> iterations. For examples, if accumulate_size = 4,
        then 4 iterations are considered as one effective iteration.

        Args:
            dataloader (``Iterable``): Your dataloader object for gradient accumulation.
            accumulate_size (int): The number of steps to accumulate gradients.

        """
        return len(dataloader) // accumulate_size

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.lr_scheduler, __name)

    def step(self, *args, **kwargs) -> None:
        """
        Update the learning rate.

        Args:
            *args: positional arguments for the lr scheduler wrapped.
            **kwargs: keyword arguments for the lr scheduler wrapped.
        """
        self.accumulate_step += 1
        if self.accumulate_step < self.accumulate_size:
            pass
        else:
            self.accumulate_step = 0
            self.lr_scheduler.step(*args, **kwargs)

    def get_lr(self) -> Tensor:
        """
        Compute the next learning rate.

        Returns:
            Tensor: the upcoming learning rate.
        """

        return self.lr_scheduler.get_lr()

    def get_last_lr(self) -> Tensor:
        """
        Returns the current learning rate.

        Returns:
            Tensor: the current learning rate.
        """

        return self.lr_scheduler.get_last_lr()

    def print_lr(self, *args, **kwargs) -> None:
        """
        Print he learning rate.

        Args:
            *args: positional arguments for the lr scheduler wrapped.
            **kwargs: keyword arguments for the lr scheduler wrapped.
        """
        self.lr_scheduler.print_lr(*args, **kwargs)

    def state_dict(self) -> dict:
        """
        Returns the states of the lr scheduler as dictionary.

        Returns:
            dict: the states of the lr scheduler.
        """
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the states of the lr scheduler from a dictionary object.

        Returns:
            dict: the states of the lr scheduler.
        """
        self.lr_scheduler.load_state_dict(state_dict)


class GradAccumGradientHandler:
    r"""A wrapper for the gradient handler to enable gradient accumulation by skipping the steps
    before accumulation size is reached.

    Args:
        grad_handler (:class:`colossalai.legacy.engine.BaseGradientHandler`):
            Your ``gradient_handler`` object for gradient accumulation, would be called when achieving `accumulate_size`.
        accumulate_size (int): The number of steps to accumulate gradients.

    More details about ``gradient_handlers`` could be found in
    `Gradient_handler <https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/engine/gradient_handler>`_.

    """

    def __init__(self, grad_handler: BaseGradientHandler, accumulate_size: int) -> None:
        assert isinstance(
            grad_handler, BaseGradientHandler
        ), f"expected grad_handler to be type BaseGradientHandler, but got {type(grad_handler)}"
        self.grad_handler = grad_handler
        self.accumulate_size = accumulate_size
        self.accumulate_step = 0

    def handle_gradient(self) -> None:
        """
        Handle gradients reduction only in the last gradient accumulation step.
        """

        self.accumulate_step += 1
        if self.accumulate_step < self.accumulate_size:
            pass
        else:
            self.accumulate_step = 0
            self.grad_handler.handle_gradient()
