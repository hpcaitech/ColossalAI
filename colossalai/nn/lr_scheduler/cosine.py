from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR

from .delayed import DelayerScheduler, WarmupDelayerScheduler, WarmupScheduler


class CosineAnnealingLR(_CosineAnnealingLR):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        eta_min (int, optional): Minimum learning rate, defaults to 0.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, total_steps: int, eta_min: int = 0, last_epoch: int = -1, **kwargs):
        super().__init__(optimizer, total_steps, eta_min=eta_min, last_epoch=last_epoch)


class CosineAnnealingWarmupLR(WarmupScheduler):
    """Cosine annealing learning rate scheduler with learning rate warmup. A linear warmup schedule will be applied.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0.
        eta_min (int, optional): Minimum learning rate, defaults to 0.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 0, eta_min: float = 0.0, last_epoch: int = -1):
        base_scheduler = _CosineAnnealingLR(
            optimizer, total_steps - warmup_steps, eta_min=eta_min, last_epoch=last_epoch
        )
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


class FlatAnnealingLR(DelayerScheduler):
    """Flat and cosine annealing learning rate scheduler. The learning rate will be a fixed value before starting decay.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        pct_start (float, optional): Percent of steps before starting learning rate decay, defaults to -0.72.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, total_steps: int, pct_start: float = 0.72, last_epoch: int = -1, **kwargs):
        if not (0.0 <= pct_start <= 1.0):
            raise ValueError(f"pct_start must >= 0.0 and <= 1.0, got {pct_start}")
        flat_steps = int(total_steps * pct_start)
        anneal_steps = total_steps - flat_steps
        base_scheduler = _CosineAnnealingLR(optimizer, anneal_steps)
        super().__init__(optimizer, flat_steps, base_scheduler, last_epoch=last_epoch)


class FlatAnnealingWarmupLR(WarmupDelayerScheduler):
    """Flat and cosine annealing learning rate scheduler with learning rate warmup. A linear warmup schedule will be
    applied, and then the learning rate will be a fixed value before starting decay.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0.
        pct_start (float, optional): Percent of steps before starting learning rate decay, defaults to -0.72.
        eta_min (int, optional): Minimum learning rate, defaults to 0.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        pct_start: float = 0.72,
        eta_min: int = 0,
        last_epoch: int = -1,
        **kwargs,
    ):
        if not (0.0 <= pct_start <= 1.0):
            raise ValueError(f"pct_start must >= 0.0 and <= 1.0, got {pct_start}")
        flat_steps = int((total_steps - warmup_steps) * pct_start)
        anneal_steps = total_steps - warmup_steps - flat_steps
        base_scheduler = _CosineAnnealingLR(optimizer, anneal_steps, eta_min=eta_min)
        super().__init__(optimizer, warmup_steps, flat_steps, base_scheduler, last_epoch=last_epoch)
