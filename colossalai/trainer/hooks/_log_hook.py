#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import os.path as osp

import torch
from tensorboardX import SummaryWriter

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.registry import HOOKS
from colossalai.trainer._trainer import Trainer
from colossalai.utils import get_global_multitimer, set_global_multitimer_status, report_memory_usage, is_dp_rank_0, \
    is_tp_rank_0, is_no_pp_or_last_stage
from ._metric_hook import MetricHook


def _format_number(val):
    if isinstance(val, float):
        return f'{val:.5f}'
    elif torch.is_floating_point(val):
        return f'{val.item():.5f}'
    return val


class EpochIntervalHook(MetricHook):
    def __init__(self, trainer: Trainer, interval: int = 1, priority: int = 1):
        super().__init__(trainer, priority)
        self._interval = interval

    def _is_epoch_to_log(self):
        return self.trainer.cur_epoch % self._interval == 0


@HOOKS.register_module
class LogMetricByEpochHook(EpochIntervalHook):
    """Specialized Hook to record the metric to log.

    :param trainer: Trainer attached with current hook
    :type trainer: Trainer
    :param interval: Recording interval
    :type interval: int, optional
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type priority: int, optional
    """

    def __init__(self, trainer: Trainer, interval: int = 1, priority: int = 1) -> None:
        super().__init__(trainer=trainer, interval=interval, priority=priority)
        self._is_rank_to_log = is_dp_rank_0() and is_tp_rank_0() and is_no_pp_or_last_stage()

    def _get_str(self, mode):
        msg = []
        for metric_name, metric_calculator in self.trainer.states['metrics'][mode].items():
            msg.append(
                f'{metric_name} = {_format_number(metric_calculator.get_accumulated_value())}')
        msg = ', '.join(msg)
        return msg

    def after_train_epoch(self):
        if self._is_epoch_to_log():
            msg = self._get_str(mode='train')

            if self._is_rank_to_log:
                self.logger.info(
                    f'Training - Epoch {self.trainer.cur_epoch} - {self.__class__.__name__}: {msg}')

    def after_test_epoch(self):
        if self._is_epoch_to_log():
            msg = self._get_str(mode='test')
            if self._is_rank_to_log:
                self.logger.info(
                    f'Testing - Epoch {self.trainer.cur_epoch} - {self.__class__.__name__}: {msg}')


@HOOKS.register_module
class TensorboardHook(MetricHook):
    """Specialized Hook to record the metric to Tensorboard.

    :param trainer: Trainer attached with current hook
    :type trainer: Trainer
    :param log_dir: Directory of log
    :type log_dir: str, optional
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type priority: int, optional
    """

    def __init__(self, trainer: Trainer, log_dir: str, priority: int = 1) -> None:
        super().__init__(trainer=trainer, priority=priority)
        self._is_rank_to_log = is_no_pp_or_last_stage()

        if self._is_rank_to_log:
            # create workspace on only one rank
            if gpc.is_initialized(ParallelMode.GLOBAL):
                rank = gpc.get_global_rank()
            else:
                rank = 0

            log_dir = osp.join(log_dir, f'rank_{rank}')

            # create workspace
            if not osp.exists(log_dir):
                os.makedirs(log_dir)

            self.writer = SummaryWriter(
                log_dir=log_dir, filename_suffix=f'_rank_{rank}')

    def after_train_iter(self, *args):
        for metric_name, metric_calculator in self.trainer.states['metrics']['train'].items():
            if metric_calculator.epoch_only:
                continue
            val = metric_calculator.get_last_step_value()
            if self._is_rank_to_log:
                self.writer.add_scalar(
                    f'{metric_name}/train', val, self.trainer.cur_step)

    def after_test_iter(self, *args):
        for metric_name, metric_calculator in self.trainer.states['metrics']['test'].items():
            if metric_calculator.epoch_only:
                continue
            val = metric_calculator.get_last_step_value()
            if self._is_rank_to_log:
                self.writer.add_scalar(f'{metric_name}/test', val,
                                       self.trainer.cur_step)

    def after_test_epoch(self):
        for metric_name, metric_calculator in self.trainer.states['metrics']['test'].items():
            if metric_calculator.epoch_only:
                val = metric_calculator.get_accumulated_value()
                if self._is_rank_to_log:
                    self.writer.add_scalar(f'{metric_name}/test', val,
                                           self.trainer.cur_step)

    def after_train_epoch(self):
        for metric_name, metric_calculator in self.trainer.states['metrics']['train'].items():
            if metric_calculator.epoch_only:
                val = metric_calculator.get_accumulated_value()
                if self._is_rank_to_log:
                    self.writer.add_scalar(f'{metric_name}/train', val,
                                           self.trainer.cur_step)


@HOOKS.register_module
class LogTimingByEpochHook(EpochIntervalHook):
    """Specialized Hook to write timing record to log.

    :param trainer: Trainer attached with current hook
    :type trainer: Trainer
    :param interval: Recording interval
    :type interval: int, optional
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type priority: int, optional
    :param log_eval: Whether writes in evaluation
    :type log_eval: bool, optional
    """

    def __init__(self,
                 trainer: Trainer,
                 interval: int = 1,
                 priority: int = 1,
                 log_eval: bool = True
                 ) -> None:
        super().__init__(trainer=trainer, interval=interval, priority=priority)
        set_global_multitimer_status(True)
        self._global_timer = get_global_multitimer()
        self._log_eval = log_eval
        self._is_rank_to_log = is_dp_rank_0() and is_tp_rank_0()

    def _get_message(self):
        msg = []
        for timer_name, timer in self._global_timer:
            last_elapsed_time = timer.get_elapsed_time()
            if timer.has_history:
                history_mean = timer.get_history_mean()
                history_sum = timer.get_history_sum()
                msg.append(
                    f'{timer_name}: last elapsed time = {last_elapsed_time}, '
                    f'history sum = {history_sum}, history mean = {history_mean}')
            else:
                msg.append(
                    f'{timer_name}: last elapsed time = {last_elapsed_time}')

        msg = ', '.join(msg)
        return msg

    def after_train_epoch(self):
        """Writes log after finishing a training epoch.
        """
        if self._is_epoch_to_log() and self._is_rank_to_log:
            msg = self._get_message()
            self.logger.info(
                f'Training - Epoch {self.trainer.cur_epoch} - {self.__class__.__name__}: {msg}')

    def after_test_epoch(self):
        """Writes log after finishing a testing epoch.
        """
        if self._is_epoch_to_log() and self._is_rank_to_log and self._log_eval:
            msg = self._get_message()
            self.logger.info(
                f'Testing - Epoch {self.trainer.cur_epoch} - {self.__class__.__name__}: {msg}')


@HOOKS.register_module
class LogMemoryByEpochHook(EpochIntervalHook):
    """Specialized Hook to write memory usage record to log.

    :param trainer: Trainer attached with current hook
    :type trainer: Trainer
    :param interval: Recording interval
    :type interval: int, optional
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type priority: int, optional
    :param log_eval: Whether writes in evaluation
    :type log_eval: bool, optional
    """

    def __init__(self,
                 trainer: Trainer,
                 interval: int = 1,
                 priority: int = 1,
                 log_eval: bool = True
                 ) -> None:
        super().__init__(trainer=trainer, interval=interval, priority=priority)
        set_global_multitimer_status(True)
        self._global_timer = get_global_multitimer()
        self._log_eval = log_eval
        self._is_rank_to_log = is_dp_rank_0() and is_tp_rank_0()

    def before_train(self):
        """Resets before training.
        """
        if self._is_epoch_to_log() and self._is_rank_to_log:
            report_memory_usage('before-train')

    def after_train_epoch(self):
        """Writes log after finishing a training epoch.
        """
        if self._is_epoch_to_log() and self._is_rank_to_log:
            report_memory_usage(
                f'After Train - Epoch {self.trainer.cur_epoch} - {self.__class__.__name__}')

    def after_test(self):
        """Reports after testing.
        """
        if self._is_epoch_to_log() and self._is_rank_to_log and self._log_eval:
            report_memory_usage(
                f'After Test - Epoch {self.trainer.cur_epoch} - {self.__class__.__name__}')
