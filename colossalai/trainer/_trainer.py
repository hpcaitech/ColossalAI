#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Union, List
from colossalai import engine
from colossalai.context.parallel_mode import ParallelMode

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from colossalai.core import global_context as gpc
from colossalai.engine import Engine
from colossalai.engine.schedule import NonPipelineSchedule, BaseSchedule
from colossalai.logging import DistributedLogger
from colossalai.utils import MultiTimer
from colossalai.utils import is_dp_rank_0, is_tp_rank_0, is_no_pp_or_last_stage
from .hooks import BaseHook


class Trainer:
    """This a class tending for easy deployments of users' training and evaluation instead of 
    writing their own scripts. It is similar with ``ignite.engine`` and ``keras.engine``, but is 
    called `Trainer`.

    :param engine: Engine responsible for the process function
    :param hooks_cfg: The configuration of hooks
    :param verbose: If True, additional information will be printed
    :type engine: Engine
    :type hoooks_cfg: Config, optional
    :type verbose: bool, optional
    """

    def __init__(self,
                 engine: Engine,
                 schedule: BaseSchedule = None,
                 timer: MultiTimer = None,
                 logger: DistributedLogger = None):
        # training-ralated params
        self._engine = engine
        self._max_epochs = 0
        self._cur_epoch = 0
        self._max_steps = 0
        self._cur_step = 0
        self._steps_per_epoch = 0

        # misc params
        self._logger = logger
        self._verbose = logger is not None

        # hooks can store states in this dict, and could be consumed by other hooks
        self.states = dict()

        # build hooks
        self.hooks = list()

        # multi-timer for time benchmarking
        self._timer = timer

        # set schedule which specifies the training iteration for the engine
        if schedule is None:
            schedule = NonPipelineSchedule()
        if gpc.is_initialized(ParallelMode.PIPELINE) and gpc.get_world_size(ParallelMode.PIPELINE) > 1:
            assert not isinstance(schedule, NonPipelineSchedule), \
                'NonPipelineSchedule cannot be used for pipeline parallel training, please use PipelineSchedule instead.'
        self._schedule = schedule
        self._schedule.pre_processing(engine)

    @property
    def cur_epoch(self):
        """Returns the index of the current epoch.
        """
        return self._cur_epoch

    @cur_epoch.setter
    def cur_epoch(self, epoch: int):
        """Set how many epochs have been processed.
        """
        # allow setter for training resumption
        self._cur_epoch = epoch

    @property
    def cur_step(self):
        """Returns how many iteration steps have been processed.
        """
        return self._cur_step

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def steps_per_epoch(self):
        return self._steps_per_epoch

    @property
    def engine(self):
        return self._engine

    @property
    def schedule(self):
        return self._schedule

    def _set_current_step(self, epoch: int):
        """Sets current step number.

        :param epoch: Step number to be set
        :type epoch: int
        """
        self._cur_step = epoch * self._steps_per_epoch

    def _call_timer(self, action: str, item: str, *args, **kwargs) -> None:
        """Call timer funciton with a given timer name.

        :param action: Function to be called on timer
        :type action: str
        :param item: Name of the timer
        :type item: str
        """

        if self._timer is not None:
            getattr(self._timer, action)(item, *args, **kwargs)

    def _reset_states(self) -> None:
        """Clear trainer states
        """
        self.states = dict()

    def _call_hooks(self, func, output=None):
        """Calls specific hooks in the current time point.

        :param func: A string represents the time point
        :param output: Output of the model after running a iteration or None in any other time points
        :type func: str
        :type output: optional
        """
        # Only after iter hook will receive output
        for hook in self.hooks:
            if output is None:
                getattr(hook, func)(self)
            else:
                getattr(hook, func)(self, *output)

    @staticmethod
    def _should_display_progress(display_progress: bool):
        """ Only display progress on DP rank 0, TP rank 0 and PP last rank
        """
        return display_progress and is_dp_rank_0() and is_tp_rank_0() and is_no_pp_or_last_stage()

    def _train_epoch(self,
                     train_dataloader: DataLoader,
                     epoch: int = None,
                     display_progress: bool = False,
                     return_output_label: bool = True):
        # set training state
        self._engine.train()
        data_iter = iter(train_dataloader)
        progress = range(self._steps_per_epoch)
        if display_progress:
            if epoch is None:
                progress = tqdm(progress, desc='[Train]')
            else:
                progress = tqdm(progress, desc=f'[Epoch {epoch} / Train]')

        self._call_hooks('before_train_epoch')
        self._call_timer(action='start', item='Train-epoch')
        for i in progress:
            self._call_hooks('before_train_iter')
            self._call_timer(action='start', item='Train-step')

            # run 1 training step
            self.engine.zero_grad()
            logits, label, loss = self.schedule.forward_backward_step(
                self.engine, data_iter, forward_only=False, return_loss=True, return_output_label=return_output_label)
            self.engine.step()
            self._call_timer(action='stop', item='Train-step', keep_in_history=True)
            self._call_hooks('after_train_iter', output=(logits, label, loss))

            self._cur_step += 1

            if display_progress:
                if 'step_metrics' in self.states:
                    progress.set_postfix(**self.states['step_metrics'])

            # stop when max iter is reached
            if self._exceed_max_step():
                break

        self._call_timer(action='stop', item='Train-epoch', keep_in_history=True)
        self._call_hooks('after_train_epoch')
        self._call_timer(action='reset', item='Train-step')

    def _eval(self,
              test_dataloader: DataLoader,
              epoch: int = None,
              display_progress: bool = False,
              return_output_label: bool = True):
        # switch engine status
        self._engine.eval()

        data_iter = iter(test_dataloader)
        num_steps = len(test_dataloader)

        self._call_hooks('before_test')
        # prepare progress bar
        progress = range(num_steps)
        if display_progress:
            desc = 'Evaluation'
            if epoch is not None:
                desc = '[Epoch %d / Test]' % epoch
            progress = tqdm(progress, desc=desc)

        self._call_hooks('before_test_epoch')
        self._call_timer(action='start', item='Test-epoch')
        with torch.no_grad():
            for _ in progress:
                self._call_hooks('before_test_iter')
                self._call_timer(action='start', item='Test-step')
                logits, label, loss = self.schedule.forward_backward_step(
                    self.engine, data_iter, forward_only=True, return_loss=True, return_output_label=return_output_label)
                self._call_timer(action='stop', item='Test-step', keep_in_history=True)
                self._call_hooks('after_test_iter',
                                 output=(logits, label, loss))

                if display_progress:
                    if 'step_metrics' in self.states:
                        progress.set_postfix(**self.states['step_metrics'])

        self._call_timer(action='stop', item='Test-epoch', keep_in_history=True)
        self._call_hooks('after_test_epoch')
        self._call_hooks('after_test')
        self._call_timer(action='reset', item='Test-step')
        self._call_timer(action='reset', item='Test-epoch')

    def _exceed_max_step(self):
        return self._max_steps is not None and self._cur_step >= self._max_steps

    def fit(self,
            train_dataloader: DataLoader,
            epochs: int,
            max_steps: int = None,
            test_dataloader: DataLoader = None,
            test_interval: int = 1,
            hooks: List[BaseHook] = None,
            display_progress: bool = False,
            return_output_label: bool = True,
            ):
        """Trains the model to fit training data.

        :param train_dataloader: DataLoader in training
        :param epochs: Maximum number of epoches
        :param max_steps: Maximum number of running iterations
        :param test_dataloader: DataLoader in testing
        :param test_interval: Interval of testing
        :param hooks_cfg: A list of hook configuration
        :param display_progress: If True, the training progress will be printed
        :param return_output_label: If True, the output of model and the label will be returned
        :type return_output_label: bool
        :type train_dataloader: DataLoader
        :type epochs: int
        :type max_steps: int
        :type test_dataloader: DataLoader
        :type test_interval: int
        :type hooks_cfg: dict
        :type display_progress: bool
        :type gradient_accumulation: int
        """

        # set epochs and steps, consider gradient accumulation
        self._steps_per_epoch = len(train_dataloader)
        self._max_steps = max_steps
        self._max_epochs = epochs

        # check if testing is required
        should_test = False
        if test_dataloader is not None:
            should_test = True

        display_progress = self._should_display_progress(display_progress)

        # reset hooks
        self._reset_states()
        if hooks is not None:
            assert isinstance(hooks, list), f'expected argument hooks be to list, but got {type(hooks)}'
        else:
            hooks = []
        self.hooks = hooks
        self.hooks.sort(key=lambda hook: hook.priority)
        if self._verbose:
            for hook in self.hooks:
                self._logger.info(
                    f'Using {hook.__class__.__name__} for training, priority = {hook.priority}', ranks=[0])
            self._logger.info("Lower value means higher priority for calling hook function", ranks=[0])
        self._call_hooks('after_hook_is_attached')

        # start train
        self._engine.train()
        self._call_hooks('before_train')

        # recover step value if resuming training
        last_epoch = self._cur_epoch
        if self.cur_epoch != 0:
            self._set_current_step(last_epoch)

        for epoch in range(last_epoch, epochs):
            # train for one epoch
            self._train_epoch(
                train_dataloader=train_dataloader,
                epoch=epoch,
                display_progress=display_progress,
                return_output_label=return_output_label
            )

            # start eval
            if should_test and epoch % test_interval == 0:
                self._eval(test_dataloader=test_dataloader,
                           display_progress=display_progress,
                           epoch=epoch,
                           return_output_label=return_output_label
                           )

            self._cur_epoch += 1

            # check for termination
            if self._exceed_max_step():
                self._logger.info(
                    f"Max number of steps {max_steps} has been reached, training is stopped automatically",
                    ranks=[0])
                break
        self._call_hooks('after_train')
        self._call_timer('reset', 'Train-epoch')

    def evaluate(self,
                 test_dataloader: DataLoader,
                 hooks: List[BaseHook] = None,
                 display_progress: bool = False,
                 return_output_label: bool = True):
        """Evaluates the model with testing data.

        :param test_dataloader: DataLoader in testing
        :param display_progress: If True, the evaluation progress will be printed
        :param return_output_label: If True, the output of model and the label will be returned
        :type test_dataloader: DataLoader
        :type display_progress: bool, optional
        :type return_output_label: bool
        """
        # set display
        display_progress = self._should_display_progress(display_progress)

        # reset hooks
        self._reset_states()
        if hooks is not None:
            assert isinstance(hooks, list), f'expected argument hooks be to list, but got {type(hooks)}'
        else:
            hooks = []
        self.hooks = hooks
        self.hooks.sort(key=lambda hook: hook.priority)
        if self._verbose:
            for hook in self.hooks:
                self._logger.info(
                    f'Using {hook.__class__.__name__} for training, priority = {hook.priority}', ranks=[0])
            self._logger.info("Lower value means higher priority for calling hook function", ranks=[0])
        self._call_hooks('after_hook_is_attached')

        # eval
        self._eval(test_dataloader=test_dataloader,
                   display_progress=display_progress,
                   return_output_label=return_output_label
                   )

    def predict(self, data: Union[Tensor, List[Tensor]]):
        """Uses trained model to make a prediction for a tensor or a tensor list.

        :param data: Data as the input
        :type data: Union[Tensor, List[Tensor]
        :return: The output of model as the prediction
        :rtype: Tensor
        """
        # predict without labels
        if isinstance(data, (list, tuple)):
            assert isinstance(data[0], Tensor)
        else:
            assert isinstance(data, Tensor)
        self._engine.eval()

        # prepare a list of (data, label) to make it iterable
        # for compatibility with schedule
        simple_dataloader = [(data, None)]
        data_iter = iter(simple_dataloader)
        output, _, _ = self.schedule.forward_backward_step(
            self.engine, data_iter, forward_only=True, return_loss=False)
        return output
