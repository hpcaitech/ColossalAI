from torch.optim.lr_scheduler import ExponentialLR as _ExponentialLR
from torch.optim.lr_scheduler import LambdaLR as _LambdaLR
from torch.optim.lr_scheduler import MultiplicativeLR as _MultiplicativeLR
from torch.optim.lr_scheduler import StepLR as _StepLR


class LambdaLR(_LambdaLR):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        lr_lambda (Union[``function``, ``list[function]``]): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such functions,
            one for each group in optimizer.param_groups, defaults to None.
        last_epoch (int, optional): The index of last epoch, defaults to -1.
    """

    def __init__(self, optimizer, total_steps, lr_lambda=None, last_epoch: int = -1) -> None:
        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class MultiplicativeLR(_MultiplicativeLR):
    """Multiply the learning rate of each parameter group by the factor given
    in the specified function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        lr_lambda (Union[``function``, ``list[function]``]): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such functions,
            one for each group in optimizer.param_groups, defaults to None.
        last_epoch (int, optional): The index of last epoch, defaults to -1.
    """

    def __init__(self, optimizer, total_steps, lr_lambda=None, last_epoch: int = -1) -> None:
        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class StepLR(_StepLR):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        step_size (int, optional): Period of learning rate decay, defaults to 1.
        gamma (float, optional): Multiplicative factor of learning rate decay, defaults to 0.1.
        last_epoch (int, optional): The index of last epoch, defaults to -1.
    """

    def __init__(self, optimizer, total_steps, step_size: int = 1, gamma: float = 0.1, last_epoch: int = -1) -> None:
        super().__init__(optimizer, step_size, gamma=gamma, last_epoch=last_epoch)


class ExponentialLR(_ExponentialLR):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr

    Args:
        optimizer (Union[:class:`torch.optim.Optimizer`, :class:`colossalai.nn.optimizer`]): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        gamma (float, optional): Multiplicative factor of learning rate decay, defaults to 1.0.
        last_epoch (int, optional): The index of last epoch, defaults to -1.
    """

    def __init__(self, optimizer, total_steps, gamma: float = 1.0, last_epoch: int = -1) -> None:
        super().__init__(optimizer, gamma, last_epoch=last_epoch)
