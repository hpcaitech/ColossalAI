from cgitb import Hook
from colossalai.registry import HOOKS
from torch import Tensor
from colossalai.trainer.hooks import BaseHook
from ._metric_hook import LearningRateMetric, MetricHook

@HOOKS.register_module
class MemTraceHook(BaseHook):
    """This class allows users to trace memory"""
    def __init__(
        self,
        truncate_by_epoch: bool = True,
        priority: int = 0,
    ) -> None:
        super().__init__(priority=priority)
        self._trunc_epoch = truncate_by_epoch

    def after_hook_is_attached(self, trainer):
        self._check_metric_states_initialization(trainer)
        trainer.states['metrics']['train']['LR'] = LearningRateMetric(epoch_only=self.by_epoch,
                                                                      initial_lr=self.lr_scheduler.get_last_lr()[0])

    def before_train(self, trainer):
        return super().before_train(trainer)
    
    def after_train(self, trainer):
        return super().after_train(trainer)
    
    def before_train_iter(self, trainer):
        return super().before_train_iter(trainer)

    def after_train_iter(self, trainer, output: Tensor, label: Tensor, loss: Tensor):
        return super().after_train_iter(trainer, output, label, loss)
    
    def after_train_epoch(self, trainer):
        if self.by_epoch:
            self.lr_scheduler.step()
            trainer.states['metrics']['train']['LR'].update(self.lr_scheduler.get_last_lr()[0])

    def after_train_iter(self, trainer, output: Tensor, label: Tensor, loss: Tensor):
        if not self.by_epoch:
            self.lr_scheduler.step()
            trainer.states['metrics']['train']['LR'].update(self.lr_scheduler.get_last_lr()[0])