from torch.optim.lr_scheduler import LambdaLR as _LambdaLR
from torch.optim.lr_scheduler import MultiplicativeLR as _MultiplicativeLR
from torch.optim.lr_scheduler import StepLR as _StepLR
from torch.optim.lr_scheduler import _LRScheduler

from colossalai.registry import LR_SCHEDULERS


@LR_SCHEDULERS.register_module
class LambdaLR(_LambdaLR):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    :param optimizer: Wrapped optimizer
    :type optimizer: torch.optim.Optimizer
    :param total_steps: number of total training steps
    :type total_steps: int
    :param lr_lambda: A function which computes a multiplicative
        factor given an integer parameter epoch, or a list of such
        functions, one for each group in optimizer.param_groups, defaults to None
    :type lr_lambda: function or list, optional
    :param num_steps_per_epoch: number of steps per epoch, defaults to -1
    :type num_steps_per_epoch: int, optional
    :param last_epoch: The index of last epoch, defaults to -1
    :type last_epoch: int, optional
    """

    def __init__(self, optimizer, total_steps, lr_lambda=None, num_steps_per_epoch: int = -1,
                 last_epoch: int = -1) -> None:
        def func(step): return lr_lambda(step // num_steps_per_epoch)

        super().__init__(optimizer, func, last_epoch=last_epoch)


@LR_SCHEDULERS.register_module
class MultiplicativeLR(_MultiplicativeLR):
    """Multiply the learning rate of each parameter group by the factor given
    in the specified function. When last_epoch=-1, sets initial lr as lr

    :param optimizer: Wrapped optimizer
    :type optimizer: torch.optim.Optimizer
    :param total_steps: number of total training steps
    :type total_steps: int
    :param lr_lambda: A function which computes a multiplicative
        factor given an integer parameter epoch, or a list of such
        functions, one for each group in optimizer.param_groups, defaults to None
    :type lr_lambda: function or list, optional
    :param num_steps_per_epoch: number of steps per epoch, defaults to -1
    :type num_steps_per_epoch: int, optional
    :param last_epoch: The index of last epoch, defaults to -1
    :type last_epoch: int, optional
    """

    def __init__(self, optimizer, total_steps, lr_lambda=None, num_steps_per_epoch: int = -1,
                 last_epoch: int = -1) -> None:
        def func(step): return lr_lambda(step // num_steps_per_epoch)

        super().__init__(optimizer, func, last_epoch=last_epoch)


@LR_SCHEDULERS.register_module
class StepLR(_StepLR):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr

    :param optimizer: Wrapped optimizer
    :type optimizer: torch.optim.Optimizer
    :param total_steps: number of total training steps
    :type total_steps: int
    :param step_size: Period of learning rate decay, defaults to 1
    :type step_size: int, optional
    :param gamma: Multiplicative factor of learning rate decay, defaults to 0.1
    :type gamma: float, optional
    :param num_steps_per_epoch: number of steps per epoch, defaults to -1
    :type num_steps_per_epoch: int, optional
    :param last_epoch: The index of last epoch, defaults to -1
    :type last_epoch: int, optional
    """

    def __init__(self, optimizer, total_steps, step_size: int = 1, gamma: float = 0.1, num_steps_per_epoch: int = -1,
                 last_epoch: int = -1) -> None:
        super().__init__(optimizer, step_size * num_steps_per_epoch,
                         gamma=gamma, last_epoch=last_epoch)


@LR_SCHEDULERS.register_module
class ExponentialLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr

    :param optimizer: Wrapped optimizer
    :type optimizer: torch.optim.Optimizer
    :param total_steps: number of total training steps
    :type total_steps: int
    :param gamma: Multiplicative factor of learning rate decay, defaults to 1.0
    :type gamma: float, optional
    :param num_steps_per_epoch: number of steps per epoch, defaults to -1
    :type num_steps_per_epoch: int, optional
    :param last_epoch: The index of last epoch, defaults to -1
    :type last_epoch: int, optional
    """

    def __init__(self, optimizer, total_steps, gamma: float = 1.0, num_steps_per_epoch: int = -1,
                 last_epoch: int = -1) -> None:
        self.gamma = gamma
        self.num_steps_per_epoch = num_steps_per_epoch
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch + 1) % self.num_steps_per_epoch == 0:
            return [group['lr'] * self.gamma
                    for group in self.optimizer.param_groups]
        return [group['lr']
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.num_steps_per_epoch)
                for base_lr in self.base_lrs]
