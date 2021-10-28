from torch.optim.lr_scheduler import _LRScheduler

from colossalai.registry import LR_SCHEDULERS
from .delayed import WarmupScheduler


@LR_SCHEDULERS.register_module
class PolynomialLR(_LRScheduler):
    """Polynomial learning rate scheduler.

    :param optimizer: Wrapped optimizer
    :type optimizer: torch.optim.Optimizer
    :param total_steps: number of total training steps
    :type total_steps: int
    :param end_lr: Minimum learning rate, defaults to 0.0001
    :type end_lr: float, optional
    :param power: the power of polynomial, defaults to 1.0
    :type power: float, optional
    :param last_epoch: The index of last epoch, defaults to -1
    :type last_epoch: int, optional
    """

    def __init__(self, optimizer, total_steps: int, end_lr: float = 0.0001, power: float = 1.0, last_epoch: int = -1,
                 **kwargs):
        if end_lr < 0:
            raise ValueError(f'end_lr must >= 0, got {end_lr}')
        self.total_steps = total_steps
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        return [
            (base_lr - self.end_lr) * ((1 - min(self.last_epoch, self.total_steps) /
                                        self.total_steps) ** self.power) + self.end_lr
            for base_lr in self.base_lrs
        ]


@LR_SCHEDULERS.register_module
class PolynomialWarmupLR(WarmupScheduler):
    """Polynomial learning rate scheduler with warmup.

    :param optimizer: Wrapped optimizer
    :type optimizer: torch.optim.Optimizer
    :param total_steps: number of total training steps
    :type total_steps: int
    :param warmup_steps: number of warmup steps, defaults to 0
    :type warmup_steps: int, optional
    :param end_lr: Minimum learning rate, defaults to 0.0001
    :type end_lr: float, optional
    :param power: the power of polynomial, defaults to 1.0
    :type power: float, optional
    :param last_epoch: The index of last epoch, defaults to -1
    :type last_epoch: int, optional
    """

    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 0, end_lr: float = 0.0001, power: float = 1.0,
                 last_epoch: int = -1, **kwargs):
        base_scheduler = PolynomialLR(
            optimizer, total_steps - warmup_steps, end_lr=end_lr, power=power)
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)
