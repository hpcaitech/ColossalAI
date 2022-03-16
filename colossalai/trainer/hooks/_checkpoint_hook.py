#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path as osp
from colossalai.logging import get_dist_logger

from colossalai.registry import HOOKS
from colossalai.trainer.hooks import BaseHook
from colossalai.utils import is_dp_rank_0
from colossalai.utils.checkpointing import get_latest_epoch, get_checkpoint_path, get_old_world_sizes, assert_divisiblity
from colossalai.utils.checkpointing import save_checkpoint, load_checkpoint
from ._lr_scheduler_hook import LRSchedulerHook


@HOOKS.register_module
class SaveCheckpointHook(BaseHook):
    """Saves the model by interval in training process.

    :param interval: Saving interval, defaults to 1
    :type interval: int, optional
    :param checkpoint_dir: Directory of saving checkpoint, defaults to None
    :type checkpoint_dir: str, optional
    :param suffix: Saving suffix of the file, defaults to ''
    :type suffix: str, optional
    :param priority: Priority in the printing, hooks with small priority will be printed in front, defaults to 10
    :type priority: int, optional
    """

    def __init__(self,
                 interval: int = 1,
                 checkpoint_dir: str = None,
                 suffix: str = '',
                 priority: int = 10):
        super().__init__(priority=priority)
        self.interval = interval
        self.checkpoint_dir = checkpoint_dir
        self.suffix = suffix
        self.logger = get_dist_logger()

        # get lr scheduler from the LRSchedulerHook before train
        self._lr_scheduler = None

    def after_hook_is_attached(self, trainer):
        # check if lr scheduler is present in LRSchedulerHook
        for hook in trainer.hooks:
            if isinstance(hook, LRSchedulerHook):
                self._lr_scheduler = hook.lr_scheduler
                break

    def after_train_epoch(self, trainer):
        """Saves the model after a training epoch.
        """
        # save by interval
        if trainer.cur_epoch % self.interval == 0:
            # only gpus with data parallel rank equals to 0 write to the disk
            if is_dp_rank_0():
                save_path = get_checkpoint_path(self.checkpoint_dir,
                                                trainer.cur_epoch,
                                                suffix=self.suffix)

                save_checkpoint(save_path,
                                trainer.cur_epoch,
                                trainer.engine.model,
                                trainer.engine.optimizer,
                                self._lr_scheduler)
                self.logger.info(
                    f'checkpoint for epoch {trainer.cur_epoch} is saved to {self.checkpoint_dir}', ranks=[0])


@HOOKS.register_module
class LoadCheckpointHook(BaseHook):
    """Loads the model before training process.

    :param checkpoint_dir: Directory of saving checkpoint, defaults to None
    :type checkpoint_dir: str, optional
    :param epoch: Epoch number to be set, defaults to -1
    :type epoch: str, optional
    :param finetune: Whether allows to load a part of the model, defaults to False
    :type finetune: bool, optional
    :param strict: Whether loads a model that has the same shape of parameters, defaults to False
    :type strict: bool, optional
    :param suffix: Suffic, defaults to ''
    :type suffix: str, optional
    :param priority: Priority in the printing, hooks with small priority will be printed in front, defaults to 0
    :type priority: int, optional
    """

    def __init__(self,
                 checkpoint_dir: str = None,
                 epoch: int = -1,
                 finetune: bool = False,
                 strict: bool = False,
                 suffix: str = '',
                 priority: int = 0) -> None:
        super().__init__(priority=priority)
        self.epoch = epoch
        self.checkpoint_dir = checkpoint_dir
        self.finetune = finetune
        self.suffix = suffix
        self.strict = strict
        self.logger = get_dist_logger()

    def before_train(self, trainer):
        """Loads parameters to the model before training.
        """
        # check if lr scheduler is present in LRSchedulerHook
        lr_scheduler = None
        for hook in trainer.hooks:
            if isinstance(hook, LRSchedulerHook):
                lr_scheduler = hook.lr_scheduler
                break
        
        old_tp_size, old_pp_size = get_old_world_sizes(self.checkpoint_dir)
        assert_divisiblity(old_tp_size, old_pp_size)
        # use latest checkpoint if epoch = -1
        if self.epoch == -1:
            self.epoch = get_latest_epoch(self.checkpoint_dir, old_tp_size, old_pp_size, suffix=self.suffix)
        
        # if osp.exists(path):
        last_epoch, _ = load_checkpoint(self.checkpoint_dir, old_tp_size, old_pp_size, self.epoch,
                                        trainer.engine.model,
                                        trainer.engine.optimizer,
                                        lr_scheduler,
                                        suffix=self.suffix,
                                        finetune=self.finetune,
                                        strict=self.strict)
        if self.finetune:
            trainer.cur_epoch = 0
        else:
            trainer.cur_epoch = last_epoch

        self.logger.info(
            f'loaded checkpoint', ranks=[0])
        # else:
        #     raise FileNotFoundError(f'checkpoint is not found at {path}')
