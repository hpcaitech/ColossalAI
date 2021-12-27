#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import os.path as osp

import torch
from typing import List
from decimal import Decimal
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.registry import HOOKS
from colossalai.logging import DistributedLogger
from colossalai.utils import report_memory_usage, is_dp_rank_0, \
    is_tp_rank_0, is_no_pp_or_last_stage, MultiTimer
from ._base_hook import BaseHook


def _format_number(val):
    if isinstance(val, float):
        return f'{val:.5g}'
    elif torch.is_tensor(val) and torch.is_floating_point(val):
        return f'{val.item():.5g}'
    return val


class LogByEpochHook(BaseHook):
    def __init__(self,
                 logger,
                 interval: int = 1,
                 priority: int = 1):
        super().__init__(priority)
        self.logger = logger
        self._interval = interval

    def _is_epoch_to_log(self, trainer):
        return trainer.cur_epoch % self._interval == 0


@HOOKS.register_module
class LogMetricByEpochHook(LogByEpochHook):
    """Specialized Hook to record the metric to log.

    :param trainer: Trainer attached with current hook
    :type trainer: Trainer
    :param interval: Recording interval
    :type interval: int, optional
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type priority: int, optional
    """

    def __init__(self,
                 logger,
                 interval: int = 1,
                 priority: int = 10) -> None:
        super().__init__(logger, interval, priority)
        self._is_rank_to_log = is_dp_rank_0() and is_tp_rank_0() and is_no_pp_or_last_stage()

    def _get_str(self, trainer, mode):
        msg = []
        for metric_name, metric_calculator in trainer.states['metrics'][mode].items():
            msg.append(
                f'{metric_name} = {_format_number(metric_calculator.get_accumulated_value())}')
        msg = ', '.join(msg)
        return msg

    def after_train_epoch(self, trainer):
        if self._is_epoch_to_log(trainer):
            msg = self._get_str(trainer=trainer, mode='train')

            if self._is_rank_to_log:
                self.logger.info(
                    f'Training - Epoch {trainer.cur_epoch} - {self.__class__.__name__}: {msg}')

    def after_test_epoch(self, trainer):
        if self._is_epoch_to_log(trainer):
            msg = self._get_str(trainer=trainer, mode='test')
            if self._is_rank_to_log:
                self.logger.info(
                    f'Testing - Epoch {trainer.cur_epoch} - {self.__class__.__name__}: {msg}')


@HOOKS.register_module
class TensorboardHook(BaseHook):
    """Specialized Hook to record the metric to Tensorboard.

    :param trainer: Trainer attached with current hook
    :type trainer: Trainer
    :param log_dir: Directory of log
    :type log_dir: str, optional
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type priority: int, optional
    """

    def __init__(self,
                 log_dir: str,
                 ranks: List = None,
                 parallel_mode: ParallelMode = ParallelMode.GLOBAL,
                 priority: int = 10,
                 ) -> None:
        super().__init__(priority=priority)
        from torch.utils.tensorboard import SummaryWriter

        # create log dir
        if not gpc.is_initialized(ParallelMode.GLOBAL) or gpc.get_global_rank() == 0:
            os.makedirs(log_dir, exist_ok=True)

        # determine the ranks to generate tensorboard logs
        self._is_valid_rank_to_log = False
        if not gpc.is_initialized(parallel_mode):
            self._is_valid_rank_to_log = True
        else:
            local_rank = gpc.get_local_rank(parallel_mode)

            if ranks is None or local_rank in ranks:
                self._is_valid_rank_to_log = True

        # check for
        if gpc.is_initialized(ParallelMode.PIPELINE) and \
                not gpc.is_last_rank(ParallelMode.PIPELINE) and self._is_valid_rank_to_log:
            raise ValueError("Tensorboard hook can only log on the last rank of pipeline process group")

        if self._is_valid_rank_to_log:
            # create workspace on only one rank
            if gpc.is_initialized(parallel_mode):
                rank = gpc.get_local_rank(parallel_mode)
            else:
                rank = 0

            # create workspace
            log_dir = osp.join(log_dir, f'{parallel_mode}_rank_{rank}')
            os.makedirs(log_dir, exist_ok=True)

            self.writer = SummaryWriter(
                log_dir=log_dir, filename_suffix=f'_rank_{rank}')

    def _log_by_iter(self, trainer, mode: str):
        for metric_name, metric_calculator in trainer.states['metrics'][mode].items():
            if metric_calculator.epoch_only:
                continue
            val = metric_calculator.get_last_step_value()

            if self._is_valid_rank_to_log:
                self.writer.add_scalar(f'{metric_name}/{mode}', val,
                                       trainer.cur_step)

    def _log_by_epoch(self, trainer, mode: str):
        for metric_name, metric_calculator in trainer.states['metrics'][mode].items():
            if metric_calculator.epoch_only:
                val = metric_calculator.get_accumulated_value()
                if self._is_valid_rank_to_log:
                    self.writer.add_scalar(f'{metric_name}/{mode}', val,
                                           trainer.cur_step)

    def after_test_iter(self, trainer, *args):
        self._log_by_iter(trainer, mode='test')

    def after_test_epoch(self, trainer):
        self._log_by_epoch(trainer, mode='test')

    def after_train_iter(self, trainer, *args):
        self._log_by_iter(trainer, mode='train')

    def after_train_epoch(self, trainer):
        self._log_by_epoch(trainer, mode='train')


@HOOKS.register_module
class LogTimingByEpochHook(LogByEpochHook):
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
                 timer: MultiTimer,
                 logger: DistributedLogger,
                 interval: int = 1,
                 priority: int = 10,
                 log_eval: bool = True,
                 ignore_num_train_steps: int = 0
                 ) -> None:
        super().__init__(logger=logger, interval=interval, priority=priority)
        self._timer = timer
        self._log_eval = log_eval
        self._is_rank_to_log = is_dp_rank_0() and is_tp_rank_0()

        # extra handling to avoid the unstable readings of the first
        # few training steps to affect the history mean time
        self._ignore_num_train_steps = ignore_num_train_steps
        self._is_train_step_history_trimmed = False

    def _get_message(self):
        msg = []
        for timer_name, timer in self._timer:
            last_elapsed_time = timer.get_elapsed_time()
            if timer.has_history:
                if timer_name == 'train-step' and not self._is_train_step_history_trimmed:
                    timer._history = timer._history[self._ignore_num_train_steps:]
                    self._is_train_step_history_trimmed = True
                history_mean = timer.get_history_mean()
                history_sum = timer.get_history_sum()
                msg.append(
                    f'{timer_name}: last = {_format_number(last_elapsed_time)} s, mean = {_format_number(history_mean)} s')
            else:
                msg.append(
                    f'{timer_name}: last = {_format_number(last_elapsed_time)} s')

        msg = ', '.join(msg)
        return msg

    def after_train_epoch(self, trainer):
        """Writes log after finishing a training epoch.
        """
        if self._is_epoch_to_log(trainer) and self._is_rank_to_log:
            msg = self._get_message()
            self.logger.info(
                f'Training - Epoch {trainer.cur_epoch} - {self.__class__.__name__}: {msg}, num steps per epoch={trainer.steps_per_epoch}')

    def after_test_epoch(self, trainer):
        """Writes log after finishing a testing epoch.
        """
        if self._is_epoch_to_log(trainer) and self._is_rank_to_log and self._log_eval:
            msg = self._get_message()
            self.logger.info(
                f'Testing - Epoch {trainer.cur_epoch} - {self.__class__.__name__}: {msg}')


@HOOKS.register_module
class LogMemoryByEpochHook(LogByEpochHook):
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
                 logger: DistributedLogger,
                 interval: int = 1,
                 priority: int = 10,
                 log_eval: bool = True,
                 report_cpu: bool = False
                 ) -> None:
        super().__init__(logger=logger, interval=interval, priority=priority)
        self._log_eval = log_eval
        self._is_rank_to_log = is_dp_rank_0() and is_tp_rank_0()

    def before_train(self, trainer):
        """Resets before training.
        """
        if self._is_epoch_to_log(trainer) and self._is_rank_to_log:
            report_memory_usage('before-train', self.logger)

    def after_train_epoch(self, trainer):
        """Writes log after finishing a training epoch.
        """
        if self._is_epoch_to_log(trainer) and self._is_rank_to_log:
            report_memory_usage(
                f'After Train - Epoch {trainer.cur_epoch} - {self.__class__.__name__}',
                self.logger)

    def after_test(self, trainer):
        """Reports after testing.
        """
        if self._is_epoch_to_log(trainer) and self._is_rank_to_log and self._log_eval:
            report_memory_usage(
                f'After Test - Epoch {trainer.cur_epoch} - {self.__class__.__name__}',
                self.logger)
