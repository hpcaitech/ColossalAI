from torch import Tensor

from colossalai.builder import build_lr_scheduler
from colossalai.registry import HOOKS
from ._metric_hook import MetricHook
from .._trainer import Trainer
from ..metric import LearningRate


@HOOKS.register_module
class LRSchedulerHook(MetricHook):
    """Build LR scheduler

    :param trainer: Trainer attached with current hook
    :type trainer: Trainer
    :param lr_scheduler_cfg: The config of LR scheduler
    :type lr_scheduler_cfg: dict
    :param by_epoch: If `True`, the LR will be scheduled every epoch. Else, the LR will be scheduled every batch. Defaults to `True`.
    :type by_epoch: bool
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type priority: int, optional
    """

    def __init__(self,
                 trainer: Trainer,
                 lr_scheduler_cfg: dict,
                 by_epoch: bool = True,
                 store_lr_in_state: bool = True,
                 priority: int = 1,
                 ):
        super().__init__(trainer=trainer, priority=priority)
        self.by_epoch = by_epoch

        if by_epoch:
            total_steps = trainer.max_epochs
        else:
            total_steps = trainer.max_epochs * trainer.steps_per_epoch
            if trainer.max_steps is not None:
                total_steps = min(total_steps, trainer.max_steps)

        lr_scheduler_cfg['total_steps'] = total_steps

        self.lr_scheduler = build_lr_scheduler(
            lr_scheduler_cfg, trainer.engine.optimizer)

        if store_lr_in_state:
            self.trainer.states['metrics']['train']['lr'] = LearningRate(epoch_only=by_epoch,
                                                                         initial_lr=self.lr_scheduler.get_lr()[0])

    def after_train_epoch(self):
        if self.by_epoch:
            self.lr_scheduler.step()
            self.trainer.states['metrics']['train']['lr'].update(self.lr_scheduler.get_lr()[0])

    def after_train_iter(self, output: Tensor, label: Tensor, loss: Tensor):
        if not self.by_epoch:
            self.lr_scheduler.step()
            self.trainer.states['metrics']['train']['lr'].update(self.lr_scheduler.get_lr()[0])
