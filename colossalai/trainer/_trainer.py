#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional
from typing import Union, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from colossalai.builder import build_hooks
from colossalai.checkpointing import save_checkpoint, load_checkpoint, get_checkpoint_path
from colossalai.context import Config
from colossalai.engine import Engine
from colossalai.logging import get_global_dist_logger
from colossalai.utils import get_global_multitimer, is_dp_rank_0, is_tp_rank_0, is_no_pp_or_last_stage
from colossalai.nn.data import DataParallelSampler


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
                 hooks_cfg: Optional[Config] = None,
                 verbose: bool = False):
        # training-ralated params
        self._engine = engine
        self._max_epochs = float('inf')
        self._max_steps = float('inf')
        self._cur_epoch = 0
        self._cur_step = 0

        # data-related params
        self._train_dataloader = None
        self._test_dataloader = None

        # misc params
        self._display_progress = False
        self._logger = get_global_dist_logger()
        self._verbose = verbose

        # hooks can store states in this dict, and could be consumed by other hooks
        self.states = {}

        # build hooks
        self.hooks = list()
        if hooks_cfg is not None:
            for cfg in hooks_cfg:
                hook = build_hooks(cfg, self)
                self.hooks.append(hook)
        self.hooks.sort(key=lambda hook: hook.priority)
        if self._verbose:
            for hook in self.hooks:
                self._logger.info(
                    f'build {hook.__class__.__name__} for train, priority = {hook.priority}', ranks=[0])

        # timer
        self._timer = get_global_multitimer()

    @property
    def cur_epoch(self):
        """Returns the index of the current epoch.
        """
        return self._cur_epoch

    @property
    def cur_step(self):
        """Returns how many iteration steps have been processed.
        """
        return self._cur_step

    def call_hooks(self, func, output=None):
        """Calls specific hooks in the current time point.

        :param func: A string represents the time point
        :param output: Output of the model after running a iteration or None in any other time points
        :type func: str
        :type output: optional
        """
        # Only after iter hook will receive output
        for hook in self.hooks:
            if output is None:
                getattr(hook, func)()
            else:
                getattr(hook, func)(*output)

    def exceed_max_step(self):
        """Checks whether the trainer exceeds the maximum number of runnning iterations.
        """
        return self._cur_step >= self._max_steps

    def set_epoch(self, epoch):
        """Sets current epoch number.

        :param epoch: Epoch number to be set
        :type epoch: int
        """
        self._cur_epoch = epoch

    def _recover_steps(self):
        step = self.cur_step * self._engine.schedule.num_steps
        self._cur_step = step

    def _set_display_progress(self, display_progress: bool):
        self._display_progress = display_progress and is_dp_rank_0(
        ) and is_tp_rank_0() and is_no_pp_or_last_stage()

    def _train_epoch(self, epoch: int = None):
        # set sampler epoch
        if epoch is not None and \
                hasattr(self._engine.train_dataloader, 'sampler') and \
                isinstance(self._engine.train_dataloader.sampler, DataParallelSampler):
            self._engine.train_dataloader.sampler.set_epoch(epoch)

        self._engine.train()

        progress = range(self._engine.schedule.num_steps)
        if self._display_progress:
            if epoch is None:
                progress = tqdm(progress, desc='[Train]')
            else:
                progress = tqdm(progress, desc=f'[Epoch {epoch} train]')

        # train 1 epoch
        self.call_hooks('before_train_epoch')
        self._timer.start('train-epoch')
        for _ in progress:
            self._cur_step += 1

            self.call_hooks('before_train_iter')
            self._timer.start('train-step')
            logits, label, loss = self._engine.step()
            self._timer.stop('train-step', keep_in_history=True)
            self.call_hooks('after_train_iter', output=(logits, label, loss))

            if self.exceed_max_step():
                # stop when max iter is reached
                break
        self._timer.stop('train-epoch', keep_in_history=True)
        self.call_hooks('after_train_epoch')
        self._timer.reset('train-step')

    def _eval(self,
              epoch: int = None,
              return_loss: bool = True):
        # switch engine status
        self._engine.eval()

        self.call_hooks('before_test')
        with torch.no_grad():
            # prepare progress bar
            progress = range(self._engine.schedule.num_steps)
            if self._display_progress:
                desc = 'Evaluation'
                if epoch is not None:
                    desc = '[Epoch %d val]' % epoch
                progress = tqdm(progress, desc=desc)

            self.call_hooks('before_test_epoch')
            self._timer.start('test-epoch')
            for _ in progress:
                self.call_hooks('before_test_iter')
                self._timer.start('test-step')
                logits, label, loss = self._engine.step(
                    return_loss=return_loss)
                self._timer.stop('test-step', keep_in_history=True)
                self.call_hooks('after_test_iter',
                                output=(logits, label, loss))
            self._timer.stop('test-epoch', keep_in_history=True)
            self.call_hooks('after_test_epoch')
        self.call_hooks('after_test')
        self._timer.reset('test-step')
        self._timer.reset('test-epoch')

    def fit(self,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader = None,
            max_epochs: int = None,
            max_steps: int = None,
            test_interval: int = 1,
            display_progress: bool = False):
        """Trains the model to fit training data.

        :param train_dataloader: DataLoader in training
        :param test_dataloader: DataLoader in testing
        :param max_epochs: Maximum number of epoches
        :param max_steps: Maximum number of running iterations
        :param test_interval: Interval of testing
        :param display_progress: If True, the training progress will be printed
        :type train_dataloader: DataLoader
        :type test_dataloader: DataLoader
        :type max_epochs: int
        :type max_steps: int
        :type test_interval: int
        :type display_progress: bool
        """

        # prepare dataloaders
        self._train_dataloader = train_dataloader
        self._engine.set_dataloader(self._train_dataloader, train=True)
        self._engine.train()

        should_test = False
        if test_dataloader is not None:
            self._test_dataloader = test_dataloader
            self._engine.set_dataloader(self._test_dataloader, train=False)
            should_test = True

        # decide the
        if max_epochs is not None:
            self._max_epochs = max_epochs
        if max_steps is not None:
            self._max_steps = max_steps
        self._set_display_progress(display_progress)

        # start train
        self.call_hooks('before_train')

        # recover step value if resuming training
        if self.cur_epoch != 0:
            self._recover_steps()

        last_epoch = self._cur_epoch

        for epoch in range(last_epoch, self._max_epochs):
            self._cur_epoch += 1

            # train for one epoch
            self._train_epoch(epoch)

            # start eval
            if should_test and epoch % test_interval == 0:
                self._eval(epoch, return_loss=True)

            # check for termination
            if self.exceed_max_step():
                self._logger.info(
                    f"Max number of steps {self._max_steps} has been reached, training is stopped automatically")
                break
        self.call_hooks('after_train')
        self._timer.reset('train-epoch')

    def evaluate(self,
                 test_dataloader: DataLoader,
                 display_progress: bool = False):
        """Evaluates the model with testing data.

        :param test_dataloader: DataLoader in testing
        :param display_progress: If True, the evaluation progress will be printed
        :type test_dataloader: DataLoader
        :type display_progress: bool, optional
        """
        # set dataloader
        self._test_dataloader = test_dataloader
        self._engine.set_dataloader(self._test_dataloader, train=True)

        # set
        self._set_display_progress(display_progress)

        # eval
        self._eval(return_loss=True)

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
        self._engine.set_dataloader(simple_dataloader)
        output, _, _ = self._engine.step(return_loss=False)
        return output

    def save(self, path: str, suffix: str = ''):
        """Saves the model to a file.

        :param path: Relative path of the file
        :param suffix: Suffix of the file
        :type path: str
        :type suffix: str, optional
        """
        save_path = get_checkpoint_path(path,
                                        self._cur_epoch,
                                        suffix=suffix)
        save_checkpoint(save_path, self._cur_epoch, self._engine.get_model(),
                        self._engine.get_optimizer(),
                        self._engine.get_lr_scheduler())

    def load(self,
             path: str,
             finetune: bool = False,
             strict: bool = False):
        """Loads parameters to the model from a file.

        :param path: Relative path of the file
        :param finetune: Whether allows to load a part of the model
        :param strict: Whether loads a model that has the same shape of parameters 
        :type path: str
        :type finetune: bool, optional
        :type strict: bool, optional
        """
        last_epoch, _ = load_checkpoint(path,
                                        self._engine.get_model(),
                                        self._engine.get_optimizer(),
                                        self._engine.get_lr_scheduler(),
                                        finetune=finetune,
                                        strict=strict)
        if finetune:
            self.set_epoch(0)
        else:
            self.set_epoch(last_epoch)
