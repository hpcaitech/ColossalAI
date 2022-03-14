from cgitb import Hook
from colossalai.registry import HOOKS
from torch import Tensor
from colossalai.trainer.hooks import BaseHook
from colossalai.utils.memory_tracer import AsyncMemoryMonitor
from ._metric_hook import LearningRateMetric, MetricHook

@HOOKS.register_module
class MemTraceHook(BaseHook):
    """This class allows users to trace memory"""
    def __init__(
        self,
        priority: int = 0,
    ) -> None:
        super().__init__(priority=priority)
        self._memory_monitor = AsyncMemoryMonitor()

    def after_hook_is_attached(self, trainer):
        # Initialize the memory monitor      
        self._check_metric_states_initialization(trainer)
        trainer.states['metrics']['train']['LR'] = LearningRateMetric(epoch_only=self.by_epoch,
                                                                      initial_lr=self.lr_scheduler.get_last_lr()[0])
    
    def before_train_iter(self, trainer):
        return super().before_train_iter(trainer)

    def after_train_iter(self, trainer, output: Tensor, label: Tensor, loss: Tensor):
        trainer.states['metrics']
        return super().after_train_iter(trainer, output, label, loss)
    
    def after_train_epoch(self, trainer):
        if self.by_epoch:
            self.lr_scheduler.step()
            trainer.states['metrics']['train']['LR'].update(self.lr_scheduler.get_last_lr()[0])

    def after_train_iter(self, trainer, output: Tensor, label: Tensor, loss: Tensor):
        if not self.by_epoch:
            self.lr_scheduler.step()
            trainer.states['metrics']['train']['LR'].update(self.lr_scheduler.get_last_lr()[0])