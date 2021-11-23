#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from colossalai.context import ParallelMode
from colossalai.registry import HOOKS
from colossalai.utils import is_no_pp_or_last_stage
from ._base_hook import BaseHook
from .._trainer import Trainer
from ..metric import Loss, Accuracy2D, Accuracy, Accuracy2p5D, Accuracy3D


class MetricHook(BaseHook):
    """Specialized hook classes for :class:`Metric`. 
    Some help metric collectors initialize, reset and 
    update their states. Others are used to display and 
    record the metric.

    :param trainer: Trainer attached with current hook
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type trainer: Trainer
    :type priority: int
    """

    def __init__(self,
                 trainer: Trainer,
                 priority: int,
                 ):
        super().__init__(trainer, priority)
        self._is_stage_to_compute = is_no_pp_or_last_stage()
        self._check_metric_states_initialization()

    def _check_metric_states_initialization(self):
        if 'metrics' not in self.trainer.states:
            self.init_runner_states('metrics', dict(train={}, test={}))


@HOOKS.register_module
class LossHook(MetricHook):
    """Specialized hook class for :class:`Loss`.

    :param trainer: Trainer attached with current hook
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type trainer: Trainer
    :type priority: int, optional
    """

    def __init__(self, trainer: Trainer, priority: int = 0):
        super().__init__(trainer, priority)

        if self._is_stage_to_compute:
            self.train_loss = Loss(epoch_only=False)
            self.test_loss = Loss(epoch_only=True)

            # register the metric calculator
            self.trainer.states['metrics']['train'][
                self.train_loss.__class__.__name__] = self.train_loss
            self.trainer.states['metrics']['test'][
                self.test_loss.__class__.__name__] = self.test_loss

    def before_train_epoch(self):
        if self._is_stage_to_compute:
            self.train_loss.reset()

    def after_train_iter(self, logits, label, loss):
        if self._is_stage_to_compute:
            self.train_loss.update(loss)

    def before_test_epoch(self):
        if self._is_stage_to_compute:
            self.test_loss.reset()

    def after_test_iter(self, logits, label, loss):
        if self._is_stage_to_compute:
            self.test_loss.update(loss)


@HOOKS.register_module
class Accuracy2DHook(MetricHook):
    """Specialized hook class for :class:`Accuracy2D`.
    It acts the same as :class:`AccuracyHook`.

    :param trainer: Trainer attached with current hook
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type trainer: Trainer
    :type priority: int, optional
    """

    def __init__(self, trainer: Trainer, priority: int = 0):
        super().__init__(trainer, priority)

        if self._is_stage_to_compute:
            self.metric = Accuracy2D(epoch_only=True)

            # register the metric
            self.trainer.states['metrics']['test'][
                self.metric.__class__.__name__] = self.metric

    def before_test(self):
        if self._is_stage_to_compute:
            self.metric.reset()

    def after_test_iter(self, logits, label, *args):
        if self._is_stage_to_compute:
            self.metric.update(logits, label)


@HOOKS.register_module
class Accuracy2p5DHook(MetricHook):
    def __init__(self, trainer: Trainer, priority: int = 0):
        super().__init__(trainer, priority)

        if self._is_stage_to_compute:
            self.metric = Accuracy2p5D(epoch_only=True)

            # register the metric
            self.trainer.states['metrics']['test'][
                self.metric.__class__.__name__] = self.metric

    def before_test(self):
        if self._is_stage_to_compute:
            self.metric.reset()

    def after_test_iter(self, logits, label, *args):
        if self._is_stage_to_compute:
            self.metric.update(logits, label)


@HOOKS.register_module
class Accuracy3DHook(MetricHook):
    """Specialized hook class for :class:`Accuracy3D`.

    :param trainer: Trainer attached with current hook
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type trainer: Trainer
    :type priority: int
    """

    def __init__(self,
                 trainer: Trainer,
                 input_parallel_mode: ParallelMode,
                 weight_parallel_mode: ParallelMode,
                 priority: int = 10):
        super().__init__(trainer, priority)

        if self._is_stage_to_compute:
            self.metric = Accuracy3D(epoch_only=True,
                                     input_parallel_mode=input_parallel_mode,
                                     weight_parallel_mode=weight_parallel_mode)

            # register the metric
            self.trainer.states['metrics']['test'][
                self.metric.__class__.__name__] = self.metric

    def before_test(self):
        if self._is_stage_to_compute:
            self.metric.reset()

    def after_test_iter(self, logits, label, *args):
        if self._is_stage_to_compute:
            self.metric.update(logits, label)


@HOOKS.register_module
class AccuracyHook(MetricHook):
    """Specialized hook class for :class:`Accuracy`.

    :param trainer: Trainer attached with current hook
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type trainer: Trainer
    :type priority: int
    """

    def __init__(self, trainer: Trainer, priority: int = 0):
        super().__init__(trainer, priority)

        if self._is_stage_to_compute:
            self.metric = Accuracy(epoch_only=True)

            # register the metric
            self.trainer.states['metrics']['test'][
                self.metric.__class__.__name__] = self.metric

    def before_test(self):
        if self._is_stage_to_compute:
            self.metric.reset()

    def after_test_iter(self, logits, label, *args):
        if self._is_stage_to_compute:
            self.metric.update(logits, label)
