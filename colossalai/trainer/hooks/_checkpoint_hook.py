#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path as osp

from colossalai.registry import HOOKS
from colossalai.trainer import Trainer
from colossalai.trainer.hooks import BaseHook
from colossalai.utils import is_dp_rank_0
from colossalai.utils.checkpointing import get_latest_checkpoint_path, get_checkpoint_path
from colossalai.utils.checkpointing import save_checkpoint, load_checkpoint
from ._lr_scheduler_hook import LRSchedulerHook


@HOOKS.register_module
class SaveCheckpointHook(BaseHook):
    """Saves the model by interval in training process.

    :param trainer: Trainer attached with current hook
    :param interval: Saving interval 
    :param checkpoint_dir: Directory of saving checkpoint 
    :param suffix: Saving suffix of the file
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type trainer: Trainer
    :type interval: int, optional
    :type checkpoint_dir: int, optional
    :type suffix: str, optional
    :type priority: int, optional
    """

    def __init__(self,
                 trainer: Trainer,
                 interval: int = 1,
                 checkpoint_dir: str = None,
                 suffix: str = '',
                 priority: int = 10):
        super().__init__(trainer=trainer, priority=priority)
        assert isinstance(trainer, Trainer), \
            f'SaveCheckpointHook expects a Trainer, got {type(trainer)}'
        self.interval = interval
        self.checkpoint_dir = checkpoint_dir
        self.suffix = suffix

        # get lr scheduler from the LRSchedulerHook before train
        self._lr_scheduler = None

    def before_train(self):
        # check if lr scheduler is present in LRSchedulerHook
        for hook in self.trainer.hooks:
            if isinstance(hook, LRSchedulerHook):
                self._lr_scheduler = hook.lr_scheduler
                break

    def after_train_epoch(self):
        """Saves the model after a training epoch.
        """
        # save by interval
        if self.trainer.cur_epoch % self.interval == 0:
            # only gpus with data parallel rank equals to 0 write to the disk
            if is_dp_rank_0():
                save_path = get_checkpoint_path(self.checkpoint_dir,
                                                self.trainer.cur_epoch,
                                                suffix=self.suffix)

                save_checkpoint(save_path,
                                self.trainer.cur_epoch,
                                self.trainer.engine.model,
                                self.trainer.engine.optimizer,
                                self._lr_scheduler)
                self.logger.info(
                    f'checkpoint for epoch {self.trainer.cur_epoch} is saved to {self.checkpoint_dir}')


@HOOKS.register_module
class LoadCheckpointHook(BaseHook):
    """Loads the model before training process.

    :param trainer: Trainer attached with current hook
    :param checkpoint_dir: Directory of saving checkpoint 
    :param epoch: Epoch number to be set
    :param finetune: Whether allows to load a part of the model
    :param strict: Whether loads a model that has the same shape of parameters 
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type trainer: Trainer
    :type checkpoint_dir: str, optional
    :type epoch: str, optional
    :type finetune: bool, optional
    :type strict: bool, optional
    :type priority: int, optional
    """

    def __init__(self,
                 trainer: Trainer = None,
                 checkpoint_dir: str = None,
                 epoch: int = -1,
                 finetune: bool = False,
                 strict: bool = False,
                 suffix: str = '',
                 priority: int = 0) -> None:
        super().__init__(trainer=trainer, priority=priority)
        assert isinstance(trainer, Trainer), \
            f'LoadLatestCheckpointHook excepts a Trainer, got {type(trainer)}'
        self.epoch = epoch
        self.checkpoint_dir = checkpoint_dir
        self.finetune = finetune
        self.suffix = suffix
        self.strict = strict

    def before_train(self):
        """Loads parameters to the model before training.
        """
        # check if lr scheduler is present in LRSchedulerHook
        lr_scheduler = None
        for hook in self.trainer.hooks:
            if isinstance(hook, LRSchedulerHook):
                lr_scheduler = hook.lr_scheduler
                break

        # use latest checkpoint if epoch = -1
        if self.epoch == -1:
            path = get_latest_checkpoint_path(self.checkpoint_dir, suffix=self.suffix)
        else:
            path = get_checkpoint_path(self.checkpoint_dir, epoch=self.epoch, suffix=self.suffix)

        if osp.exists(path):
            last_epoch, _ = load_checkpoint(path,
                                            self.trainer.engine.model,
                                            self.trainer.engine.optimizer,
                                            lr_scheduler,
                                            finetune=self.finetune,
                                            strict=self.strict)
            if self.finetune:
                self.trainer.cur_epoch = 0
            else:
                self.trainer.cur_epoch = last_epoch

            self.logger.info(
                f'loaded checkpoint from {path}')
        else:
            raise FileNotFoundError(f'checkpoint is not found at {path}')
