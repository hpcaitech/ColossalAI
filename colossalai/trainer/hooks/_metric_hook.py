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


class Metric(ABC):
    """A basic class of metric collectors. It collects a specific
    metric during training or evaluation and it's always used with 
    :class:`MetricHook` to help it update its states and show the 
    metric. So please use corresponding hook class to make the metric 
    collector works.

    :param epoch_only: Whether the metric only read for the full epoch
    :type epoch_only: bool
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
    def get_last_step_value(self):
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

    :param epoch_only: Whether the metric only read for the full epoch
    :type epoch_only: bool
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

        :param loss: Current loss of the output
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

    def get_last_step_value(self):
        """Returns :attr:`last_step_loss`.
        """
        return self.last_step_loss

    def is_better(a, b):
        return a < b


class LearningRateMetric(Metric):
    """A metric collector for learning rate.

    :param epoch_only: Whether the metric only read for the full epoch
    :type epoch_only: bool
    :param initial_lr: initial learning rate, defaults to 0.0
    :type initial_lr: float, optional
    """

    def __init__(self, epoch_only: bool, initial_lr: float = 0.):
        super().__init__(epoch_only=epoch_only)
        self.lr = initial_lr

    def reset(self) -> None:
        pass

    def update(self, lr) -> None:
        self.lr = lr

    def get_last_step_value(self):
        return self.lr

    def get_accumulated_value(self):
        return self.lr

    def is_better(a, b) -> bool:
        pass


class AccuracyMetric(Metric):
    """A metric collector for accuracy. It only works for classification
    tasks.

    :param epoch_only: Whether the metric only read for the full epoch
    :type epoch_only: bool
    :param accuracy_func: accuracy function for the classification task
    :type accuracy_func: typing.Callable
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

        :param logits: The logits output of the model
        :param targets: real labels of the dataset
        :param batch_size: batch size of the task
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

    def get_last_step_value(self):
        self.last_step_sum = all_reduce(self.last_step_sum, ParallelMode.DATA)
        self.last_step_correct = all_reduce(self.last_step_correct, ParallelMode.DATA)
        return (self.last_step_correct / self.last_step_sum).item()

    def get_accumulated_value(self):
        self.accumulated_sum = all_reduce(self.accumulated_sum, ParallelMode.DATA)
        self.accumulated_correct = all_reduce(self.accumulated_correct, ParallelMode.DATA)
        return (self.accumulated_correct / self.accumulated_sum).item()

    def is_better(a, b) -> bool:
        return a > b


class MetricHook(BaseHook):
    """Specialized hook classes for :class:`Metric`. 
    Some help metric collectors initialize, reset and 
    update their states. Others are used to display and 
    record the metric.

    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type priority: int
    :param trainer: Trainer attached with current hook
    :type trainer: Trainer
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

    :param priority: Priority in the printing, hooks with small priority will be printed in front, defaults to 0
    :type priority: int, optional
    :param trainer: Trainer attached with current hook
    :type trainer: Trainer
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

    :param accuracy_func: Priority in the printing, hooks with small priority will be printed in front
    :type accuracy_func: typing.Callable
    :param priority: Priority in the printing, hooks with small priority will be printed in front, defaults to 0
    :type priority: int, optional
    :param trainer: Trainer attached with current hook
    :type trainer: Trainer
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
            batch_size = trainer.schedule.batch_size
            self.metric.update(logits, targets, batch_size)


class ThroughputMetric(Metric):
    """Metric for :class:`Throughput`.

    :param epoch_only: epoch only
    :type epoch_only: bool
    :param num_samples: number of samples
    :param time: time
    """
    def __init__(self, epoch_only: bool):
        super().__init__(epoch_only=epoch_only)
        self.accumulated_num_samples = torch.zeros(1, device=get_current_device())
        self.accumulated_used_time = torch.zeros(1, device=get_current_device())
        self.last_step_num_samples = torch.zeros(1, device=get_current_device())
        self.last_step_used_time = torch.zeros(1, device=get_current_device())

    def reset(self) -> None:
        self.accumulated_num_samples.zero_()
        self.accumulated_used_time.zero_()
        self.last_step_num_samples.zero_()
        self.last_step_used_time.zero_()

    def update(self, num_samples, time) -> None:
        self.last_step_num_samples.fill_(num_samples)
        self.last_step_used_time.fill_(time)
        self.accumulated_num_samples += self.last_step_num_samples
        self.accumulated_used_time += self.last_step_used_time

    def get_last_step_value(self):
        self.last_step_used_time = all_reduce(self.last_step_used_time, ParallelMode.DATA) / \
            gpc.get_world_size(ParallelMode.DATA)
        self.last_step_num_samples = all_reduce(self.last_step_num_samples, ParallelMode.DATA)
        return (self.last_step_num_samples / (self.last_step_used_time + 1e-12)).item()

    def get_accumulated_value(self):
        self.accumulated_used_time = all_reduce(self.accumulated_used_time, ParallelMode.DATA) / \
            gpc.get_world_size(ParallelMode.DATA)
        self.accumulated_num_samples = all_reduce(self.accumulated_num_samples, ParallelMode.DATA)
        return (self.accumulated_num_samples / (self.accumulated_used_time + 1e-12)).item()

    def is_better(a, b) -> bool:
        pass


@HOOKS.register_module
class ThroughputHook(MetricHook):
    """Specialized hook class for :class:`Throughput`.

    :param priority: priority of throughput hook, defaults to 10
    :type priority: int, optional
    :param trainer: Trainer attached with current hook
    :type trainer: Trainer
    """
    def __init__(self, priority: int = 10):
        super().__init__(priority)

    def after_hook_is_attached(self, trainer):
        self._check_metric_states_initialization(trainer)
        if self._is_stage_to_compute:
            self.metric = ThroughputMetric(epoch_only=True)

            # register the metric
            trainer.states['metrics']['train']['Throughput'] = self.metric
            trainer.states['metrics']['test']['Throughput'] = self.metric

    def before_train_epoch(self, trainer):
        if self._is_stage_to_compute:
            self.metric.reset()

    def after_train_iter(self, trainer, *args):
        if self._is_stage_to_compute:
            self.metric.update(trainer.schedule.batch_size, trainer._timer.get_timer('Train-step').get_elapsed_time())

    def before_test(self, trainer):
        if self._is_stage_to_compute:
            self.metric.reset()

    def after_test_iter(self, trainer, *args):
        if self._is_stage_to_compute:
            self.metric.update(trainer.schedule.batch_size, trainer._timer.get_timer('Test-step').get_elapsed_time())
