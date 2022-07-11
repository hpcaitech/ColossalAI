from torch.optim.lr_scheduler import _LRScheduler

from colossalai.registry import LR_SCHEDULERS


@LR_SCHEDULERS.register_module
class LinearWarmupLR(_LRScheduler):
    """Linearly warmup learning rate and then linearly decay.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 0, last_epoch: int = -1, **kwargs):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [(self.last_epoch + 1) / (self.warmup_steps + 1) * lr for lr in self.base_lrs]
        else:
            return [(self.total_steps - self.last_epoch) / (self.total_steps - self.warmup_steps) * lr
                    for lr in self.base_lrs]
