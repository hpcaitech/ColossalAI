#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path as osp
from colossalai.logging import get_dist_logger

from colossalai.registry import HOOKS
from colossalai.trainer.hooks import BaseHook
from colossalai.utils import is_dp_rank_0
from colossalai.utils.checkpointing import get_latest_checkpoint_path, get_checkpoint_path
from colossalai.utils.checkpointing import save_checkpoint, load_checkpoint
from ._lr_scheduler_hook import LRSchedulerHook


@HOOKS.register_module
class SaveCheckpointHook(BaseHook):
    """Saves the model by interval in training process.

    Args:
       interval (int, optional): Saving interval, defaults to 1.
       checkpoint_dir (str, optional): Directory of saving checkpoint, defaults to None.
       suffix (str, optional): Saving suffix of the file, defaults to ''.
       priority (int, optional): Priority in the printing, hooks with small priority will be printed in front
            defaults to 10. If different hooks share same priority, the order of printing would
            depend on the hooks order in the hook list.
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

    Args:
        checkpoint_dir (str, optional): Directory of saving checkpoint, defaults to None.
        epoch (str, optional): Loading checkpoint of setting epoch numbers, defaults to -1.
            Epoch equals to -1 means choosing the latest checkpoint.
        finetune (bool, optional): Whether allows to load a part of the model, defaults to False.
        strict (bool, optional): Whether to strictly enforce that the keys in :attr:`state_dict` of the checkpoint
            match the names of parameters and buffers in model, defaults to False.
        suffix (str, optional): Suffix of checkpoint file path, defaults to ''.
        priority (int, optional): Priority in the printing, hooks with small priority will be printed in front,
            defaults to 0. If different hooks share same priority, the order of printing would
            depend on the hooks order in the hook list.
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

        # use latest checkpoint if epoch = -1
        if self.epoch == -1:
            path = get_latest_checkpoint_path(self.checkpoint_dir, suffix=self.suffix)
        else:
            path = get_checkpoint_path(self.checkpoint_dir, epoch=self.epoch, suffix=self.suffix)

        if osp.exists(path):
            last_epoch, _ = load_checkpoint(path,
                                            trainer.engine.model,
                                            trainer.engine.optimizer,
                                            lr_scheduler,
                                            finetune=self.finetune,
                                            strict=self.strict)
            if self.finetune:
                trainer.cur_epoch = 0
            else:
                trainer.cur_epoch = last_epoch

            self.logger.info(
                f'loaded checkpoint from {path}', ranks=[0])
        else:
            raise FileNotFoundError(f'checkpoint is not found at {path}')
