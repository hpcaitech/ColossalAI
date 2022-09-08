from typing import List

from torch.optim.lr_scheduler import MultiStepLR as _MultiStepLR

from colossalai.registry import LR_SCHEDULERS
from .delayed import WarmupScheduler


@LR_SCHEDULERS.register_module
class MultiStepLR(_MultiStepLR):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        milestones (List[int], optional): List of epoch indices. Must be increasing, defaults to None.
        gamma (float, optional): Multiplicative factor of learning rate decay, defaults to 0.1.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self,
                 optimizer,
                 total_steps: int,
                 milestones: List[int] = None,
                 gamma: float = 0.1,
                 last_epoch: int = -1,
                 **kwargs):
        super().__init__(optimizer, milestones, gamma=gamma, last_epoch=last_epoch)


@LR_SCHEDULERS.register_module
class MultiStepWarmupLR(WarmupScheduler):
    """Multistep learning rate scheduler with warmup.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0.
        milestones (List[int], optional): List of epoch indices. Must be increasing, defaults to None.
        gamma (float, optional): Multiplicative factor of learning rate decay, defaults to 0.1.
        num_steps_per_epoch (int, optional): Number of steps per epoch, defaults to -1.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self,
                 optimizer,
                 total_steps: int,
                 warmup_steps: int = 0,
                 milestones: List[int] = None,
                 gamma: float = 0.1,
                 last_epoch: int = -1,
                 **kwargs):
        if len(milestones) == 0:
            raise ValueError('milestones cannot be empty')
        milestones = [v - warmup_steps for v in milestones if v >= warmup_steps]
        base_scheduler = _MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)
