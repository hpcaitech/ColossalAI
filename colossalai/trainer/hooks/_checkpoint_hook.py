#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from colossalai.logging import get_dist_logger

from colossalai.registry import HOOKS
from colossalai.trainer.hooks import BaseHook
from colossalai.utils.checkpointing import save_checkpoint
from ._lr_scheduler_hook import LRSchedulerHook


@HOOKS.register_module
class SaveCheckpointHook(BaseHook):
    """Saves the model by interval in training process.

    Args:
       interval (int, optional): Number of epochs between saving the checkpoint, defaults to 1.
       checkpoint_dir (str, optional): File name to save the checkpoint, defaults to None.
       priority (int, optional): Priority in the printing, hooks with small priority will be printed in front
            defaults to 10. If different hooks share same priority, the order of printing would
            depend on the hooks order in the hook list.
    """

    def __init__(self,
                 interval: int = 1,
                 checkpoint_dir: str = None,
                 priority: int = 10):
        super().__init__(priority=priority)
        self.interval = interval
        self.checkpoint_dir = checkpoint_dir
        self.logger = get_dist_logger()

        # get lr scheduler from the LRSchedulerHook before train
        self._lr_scheduler = None

    def after_hook_is_attached(self, trainer):
        # get lr scheduler if exists
        for hook in trainer.hooks:
            if isinstance(hook, LRSchedulerHook):
                self._lr_scheduler = hook.lr_scheduler
                break

    def after_train_epoch(self, trainer):
        """Saves the model after a training epoch.
        """
        # save by interval
        if trainer.cur_epoch % self.interval == 0:
            save_checkpoint(self.checkpoint_dir,
                            trainer.cur_epoch,
                            trainer.engine.model,
                            trainer.engine.optimizer,
                            self._lr_scheduler)
            self.logger.info(
                f'checkpoint for epoch {trainer.cur_epoch} is saved to {self.checkpoint_dir}', ranks=[0])
