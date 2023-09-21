from torch import Tensor

from colossalai.legacy.registry import HOOKS

from ._metric_hook import LearningRateMetric, MetricHook


@HOOKS.register_module
class LRSchedulerHook(MetricHook):
    r"""Build LR scheduler for trainer.

    Args:
        lr_scheduler (:class:`colossalai.nn.lr_scheduler`): The specific LR scheduler
            in range of ``colossalai.nn.lr_scheduler``, more details about ``lr_scheduler`` could be found in
            `lr_scheduler <https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/nn/lr_scheduler>`_.
        by_epoch (bool): If `True`, the LR will be scheduled every epoch. Else, the LR will be scheduled every batch.
        store_lr_in_state (bool, optional): If `True`, store the learning rate in each state, defaults to `True`.
        priority (int, optional): Priority in the printing, hooks with small priority will be printed in front
            defaults to 1. If different hooks share same priority, the order of printing would
            depend on the hooks order in the hook list.
    """

    def __init__(
        self,
        lr_scheduler,
        by_epoch: bool,
        store_lr_in_state: bool = True,
        priority: int = 1,
    ):
        super().__init__(priority=priority)
        self.by_epoch = by_epoch
        self.lr_scheduler = lr_scheduler
        self.store_lr_in_state = store_lr_in_state

    def after_hook_is_attached(self, trainer):
        self._check_metric_states_initialization(trainer)
        trainer.states["metrics"]["train"]["LR"] = LearningRateMetric(
            epoch_only=self.by_epoch, initial_lr=self.lr_scheduler.get_last_lr()[0]
        )

    def after_train_epoch(self, trainer):
        if self.by_epoch:
            self.lr_scheduler.step()
            trainer.states["metrics"]["train"]["LR"].update(self.lr_scheduler.get_last_lr()[0])

    def after_train_iter(self, trainer, output: Tensor, label: Tensor, loss: Tensor):
        if not self.by_epoch:
            self.lr_scheduler.step()
            trainer.states["metrics"]["train"]["LR"].update(self.lr_scheduler.get_last_lr()[0])
