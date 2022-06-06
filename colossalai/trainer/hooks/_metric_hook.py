#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.distributed as dist
from colossalai.communication import all_reduce
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.registry import HOOKS
from colossalai.utils import get_current_device, is_no_pp_or_last_stage

from ._base_hook import BaseHook
from ._commons_ import _format_number


class Metric(ABC):
    """A basic class of metric collectors. It collects a specific
    metric during training or evaluation and would always be used with
    :class:`MetricHook` to help it update its states and show the 
    metric. So please use corresponding hook class to make the metric 
    collector works.

    Args:
        epoch_only (bool): Whether the metric only read for the full epoch.
    """

    def __init__(self, epoch_only: bool):
        # is the metric only read for the full epoch
        self._epoch_only = epoch_only

    @property
    def epoch_only(self):
        """Returns :attr:`epoch_only`.
        """
        return self._epoch_only

    @abstractmethod
    def reset(self) -> None:
        """Resets the metric to it's initial state.
        By default, this is called at the start of each epoch.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Updates the metric's state using the passed batch output.
        By default, this is called once for each batch.
        """
        pass

    @abstractmethod
    def get_last_step_value(self) -> float:
        """Returns the metric value in the last iteration.
        """
        pass

    @abstractmethod
    def get_accumulated_value(self):
        """Computes the metric based on it's accumulated state.
        By default, this is called at the end of each epoch.

        :return: the actual quantity of interest
        :rtype: Any
        """
        pass

    @staticmethod
    @abstractmethod
    def is_better(a, b) -> bool:
        """Compares a and b, and returns whether a is better than b

        :return: The result of comparison
        :rtype: bool
        """
        pass


class LossMetric(Metric):
    """A metric collector for loss.

    Args:
        epoch_only (bool): Whether the metric only read for the full epoch.
    """

    def __init__(self, epoch_only):
        super().__init__(epoch_only=epoch_only)
        self.last_step_loss = torch.zeros(1, device=get_current_device())
        self.accum_loss = torch.zeros(1, device=get_current_device())
        self.count = 0

    def reset(self) -> None:
        """Sets :attr:`last_step_loss` and :attr:`accum_loss` to zero.
        """
        self.last_step_loss.zero_()
        self.accum_loss.zero_()
        self.count = 0

    def update(self, loss) -> None:
        """Updates :attr:`last_step_loss` and :attr:`accum_loss` with current loss.
        It expects the output has loss.

        Args:
            loss (:class:`torch.tensor`): Current loss of the output.
        """
        # expect output to be logits, label and loss
        loss_ = loss.detach()
        self.last_step_loss.copy_(loss_)
        self.accum_loss.add_(loss_)
        self.count += 1

    def get_accumulated_value(self):
        """Returns accumulated loss.
        """
        if gpc.is_initialized(ParallelMode.DATA):
            dist.all_reduce(self.accum_loss, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.DATA))
            self.accum_loss.div_(gpc.get_world_size(ParallelMode.DATA))

        self.accum_loss.div_(self.count)
        return self.accum_loss.item()

    def get_last_step_value(self) -> float:
        """Returns :attr:`last_step_loss`.
        """
        return self.last_step_loss.cpu().item()

    @staticmethod
    def is_better(a, b):
        return a < b


class LearningRateMetric(Metric):
    """A metric collector for learning rate.

    Args:
        epoch_only (bool): Whether the metric only read for the full epoch.
        initial_lr (float, optional): Initial learning rate, defaults to 0.0.
    """

    def __init__(self, epoch_only: bool, initial_lr: float = 0.):
        super().__init__(epoch_only=epoch_only)
        self.lr = initial_lr

    def reset(self) -> None:
        pass

    def update(self, lr) -> None:
        self.lr = lr

    def get_last_step_value(self) -> float:
        return self.lr

    def get_accumulated_value(self):
        return self.lr

    @staticmethod
    def is_better(a, b) -> bool:
        pass


class AccuracyMetric(Metric):
    """A metric collector for accuracy. It only works for classification
    tasks.

    Args:
        epoch_only (bool): Whether the metric only read for the full epoch.
        accuracy_func (:class:`typing.Callable`): Accuracy function for the classification task.
    """

    def __init__(self, epoch_only: bool, accuracy_func: Callable):
        super().__init__(epoch_only=epoch_only)
        self.acc = accuracy_func
        self.last_step_sum = torch.zeros(1, device=get_current_device())
        self.last_step_correct = torch.zeros(1, device=get_current_device())
        self.accumulated_sum = torch.zeros(1, device=get_current_device())
        self.accumulated_correct = torch.zeros(1, device=get_current_device())

    def reset(self) -> None:
        self.last_step_sum.zero_()
        self.last_step_correct.zero_()
        self.accumulated_sum.zero_()
        self.accumulated_correct.zero_()

    def update(self, logits, targets, batch_size) -> None:
        """Updates last step accuracy and accumulated accuracy with current logits
        and labels. It expects the output has logits and labels.

        Args:
            logits (:class:`torch.tensor`): The logits output of the model.
            targets (:class:`torch.tensor`): Real labels of the dataset.
            batch_size (int): Batch size of the task.
        """
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if isinstance(targets, (list, tuple)):
            targets = targets[0]
        # update
        correct = self.acc(logits, targets)

        self.last_step_sum.fill_(batch_size)
        self.last_step_correct.fill_(correct)
        self.accumulated_sum += self.last_step_sum
        self.accumulated_correct += self.last_step_correct

    def get_last_step_value(self) -> float:
        self.last_step_sum = all_reduce(self.last_step_sum, ParallelMode.DATA)
        self.last_step_correct = all_reduce(self.last_step_correct, ParallelMode.DATA)
        return _format_number((self.last_step_correct / self.last_step_sum).cpu().item())

    def get_accumulated_value(self):
        self.accumulated_sum = all_reduce(self.accumulated_sum, ParallelMode.DATA)
        self.accumulated_correct = all_reduce(self.accumulated_correct, ParallelMode.DATA)
        return (self.accumulated_correct / self.accumulated_sum).item()

    @staticmethod
    def is_better(a, b) -> bool:
        return a > b


class MetricHook(BaseHook):
    """Specialized hook classes for :class:`Metric`. 
    Some help metric collectors initialize, reset and 
    update their states. Others are used to display and 
    record the metric.

    Args:
        priority (int): Priority in the printing, hooks with small priority will be printed in front
            defaults to 1. If different hooks share same priority, the order of printing would
            depend on the hooks order in the hook list.
    """

    def __init__(
        self,
        priority: int,
    ):
        super().__init__(priority)
        self._is_stage_to_compute = is_no_pp_or_last_stage()

    def _check_metric_states_initialization(self, trainer):
        if 'metrics' not in trainer.states:
            self.init_runner_states(trainer, 'metrics', dict(train={}, test={}))


@HOOKS.register_module
class LossHook(MetricHook):
    """Specialized hook class for :class:`Loss`.

    Args:
        priority (int, optional): Priority in the printing, hooks with small priority will be printed in front
            defaults to 0. If different hooks share same priority, the order of printing would
            depend on the hooks order in the hook list.
    """

    def __init__(self, priority: int = 0):
        super().__init__(priority)

    def after_hook_is_attached(self, trainer):
        self._check_metric_states_initialization(trainer)

        if self._is_stage_to_compute:
            self.train_loss = LossMetric(epoch_only=False)
            self.test_loss = LossMetric(epoch_only=True)

            # register the metric calculator
            trainer.states['metrics']['train']['Loss'] = self.train_loss
            trainer.states['metrics']['test']['Loss'] = self.test_loss

    def before_train_epoch(self, trainer):
        if self._is_stage_to_compute:
            self.train_loss.reset()

    def after_train_iter(self, trainer, logits, label, loss):
        if self._is_stage_to_compute:
            self.train_loss.update(loss)

    def before_test_epoch(self, trainer):
        if self._is_stage_to_compute:
            self.test_loss.reset()

    def after_test_iter(self, trainer, logits, label, loss):
        if self._is_stage_to_compute:
            self.test_loss.update(loss)


@HOOKS.register_module
class AccuracyHook(MetricHook):
    """Specialized hook class for :class:`Accuracy`.

    Args:
        accuracy_func (:class:`typing.Callable`): Accuracy function for the classification task.
        priority (int, optional): Priority in the printing, hooks with small priority will be printed in front
            defaults to 0. If different hooks share same priority, the order of printing would
            depend on the hooks order in the hook list.
    """

    def __init__(self, accuracy_func: Callable, priority: int = 0):
        super().__init__(priority)
        self.accuracy_func = accuracy_func

    def after_hook_is_attached(self, trainer):
        self._check_metric_states_initialization(trainer)
        if self._is_stage_to_compute:
            self.metric = AccuracyMetric(epoch_only=True, accuracy_func=self.accuracy_func)

            # register the metric
            trainer.states['metrics']['test']['Accuracy'] = self.metric

    def before_test(self, trainer):
        if self._is_stage_to_compute:
            self.metric.reset()

    def after_test_iter(self, trainer, logits, targets, *args):
        if self._is_stage_to_compute:
            batch_size = trainer.engine.schedule.batch_size
            self.metric.update(logits, targets, batch_size)


class ThroughputMetric(Metric):
    """Metric for :class:`Throughput`.

    Args:
        epoch_only (bool): Whether the metric only read for the full epoch.
    """

    def __init__(self, epoch_only: bool, ignored_steps: int = 0, tflop_per_step: int = 0, use_local: bool = False):
        super().__init__(epoch_only=epoch_only)
        self.ignored_steps = ignored_steps
        self.cur_steps = 0
        self.accumulated_num_samples = torch.zeros(1, device=get_current_device())
        self.accumulated_used_time = torch.zeros(1, device=get_current_device())
        self.last_step_num_samples = torch.zeros(1, device=get_current_device())
        self.last_step_used_time = torch.zeros(1, device=get_current_device())
        self._tflop_per_step = tflop_per_step
        self._use_local = use_local

    def reset(self) -> None:
        # self.cur_steps = 0
        self.accumulated_num_samples.zero_()
        self.accumulated_used_time.zero_()
        self.last_step_num_samples.zero_()
        self.last_step_used_time.zero_()

    def update(self, num_samples, time) -> None:
        self.cur_steps += 1
        self.last_step_num_samples.fill_(num_samples)
        self.last_step_used_time.fill_(time)
        if self.cur_steps >= self.ignored_steps:
            self.accumulated_num_samples += self.last_step_num_samples
            self.accumulated_used_time += self.last_step_used_time

    def get_last_step_value(self) -> float:
        if self._use_local:
            self.last_step_num_samples *= gpc.get_world_size(ParallelMode.DATA)
        else:
            self.last_step_used_time = all_reduce(self.last_step_used_time, ParallelMode.DATA) / \
                 gpc.get_world_size(ParallelMode.DATA)
            self.last_step_num_samples = all_reduce(self.last_step_num_samples, ParallelMode.DATA)

        sample_per_sec = _format_number(self.last_step_num_samples / (self.last_step_used_time + 1e-12).item())
        return sample_per_sec

    def get_last_step_info(self) -> str:
        if self._use_local:
            self.last_step_num_samples *= gpc.get_world_size(ParallelMode.DATA)
        else:
            self.last_step_used_time = all_reduce(self.last_step_used_time, ParallelMode.DATA) / \
                 gpc.get_world_size(ParallelMode.DATA)
            self.last_step_num_samples = all_reduce(self.last_step_num_samples, ParallelMode.DATA)

        sample_per_sec = _format_number(self.last_step_num_samples / (self.last_step_used_time + 1e-12).item())
        if self._tflop_per_step > 0:
            tflops = _format_number(self._tflop_per_step / (self.last_step_used_time.item() + 1e-12))
            return f"{sample_per_sec} sample_per_sec, {tflops} Tflops"
        else:
            return f"{sample_per_sec} sample_per_sec"

    def get_accumulated_value(self) -> float:
        self.accumulated_used_time = all_reduce(self.accumulated_used_time, ParallelMode.DATA) / \
            gpc.get_world_size(ParallelMode.DATA)
        self.accumulated_num_samples = all_reduce(self.accumulated_num_samples, ParallelMode.DATA)
        return (self.accumulated_num_samples / (self.accumulated_used_time + 1e-12)).item()

    @staticmethod
    def is_better(a, b) -> bool:
        pass


@HOOKS.register_module
class ThroughputHook(MetricHook):
    """Specialized hook class for :class:`Throughput`. Hook to measure execution throughput (samples/sec).

    Args:
        ignored_steps (int, optional): the number of initial training steps to ignore.
        priority (int, optional): Priority in the printing, hooks with small priority will be printed in front
            defaults to 10. If different hooks share same priority, the order of printing would
            depend on the hooks order in the hook list.
        tflop_per_step(int, optional): tera floating point operations per step.
        use_local (bool, optional): Whether to use local time for throughput calculation.
    """

    def __init__(self, ignored_steps: int = 0, priority: int = 10, tflop_per_step: int = 0, use_local=False):
        super().__init__(priority)
        self.ignored_steps = ignored_steps
        self._tflop_per_step = tflop_per_step
        self._use_local = use_local

    def after_hook_is_attached(self, trainer):
        self._check_metric_states_initialization(trainer)
        if self._is_stage_to_compute:
            self.metric = ThroughputMetric(epoch_only=True,
                                           ignored_steps=self.ignored_steps,
                                           tflop_per_step=self._tflop_per_step,
                                           use_local=self._use_local)

            # register the metric
            trainer.states['metrics']['train']['Throughput'] = self.metric
            trainer.states['metrics']['test']['Throughput'] = self.metric

    def before_train_epoch(self, trainer):
        if self._is_stage_to_compute:
            self.metric.reset()

    def after_train_iter(self, trainer, *args):
        if self._is_stage_to_compute:
            self.metric.update(trainer.engine.schedule.batch_size,
                               trainer._timer.get_timer('Train-step').get_elapsed_time())

    def before_test(self, trainer):
        if self._is_stage_to_compute:
            self.metric.reset()

    def after_test_iter(self, trainer, *args):
        if self._is_stage_to_compute:
            self.metric.update(trainer.engine.schedule.batch_size,
                               trainer._timer.get_timer('Test-step').get_elapsed_time())
