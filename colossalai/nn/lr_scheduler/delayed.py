from torch.optim.lr_scheduler import _LRScheduler


class _enable_get_lr_call:
    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False


class DelayerScheduler(_LRScheduler):
    """ Starts with a flat lr schedule until it reaches N epochs the applies a scheduler 

    :param optimizer: Wrapped optimizer.
    :type optimizer: torch.optim.Optimizer
    :param delay_epochs: Number of epochs to keep the initial lr until starting aplying the scheduler
    :type delay_epochs: int
    :param after_scheduler: After target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    :type after_scheduler: torch.optim.lr_scheduler
    :param last_epoch: The index of last epoch, defaults to -1
    :type last_epoch: int, optional
    """

    def __init__(self, optimizer, delay_epochs, after_scheduler, last_epoch=-1):
        if delay_epochs < 0:
            raise ValueError(f'delay_epochs must >= 0, got {delay_epochs}')
        self.delay_epochs = delay_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            with _enable_get_lr_call(self.after_scheduler):
                return self.after_scheduler.get_lr()

        return self.base_lrs

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.delay_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(DelayerScheduler, self).step(epoch)


class WarmupScheduler(_LRScheduler):
    """ Starts with a linear warmup lr schedule until it reaches N epochs the applies a scheduler

    :param optimizer: Wrapped optimizer.
    :type optimizer: torch.optim.Optimizer
    :param warmup_epochs: Number of epochs to linearly warmup lr until starting aplying the scheduler
    :type warmup_epochs: int
    :param after_scheduler: After target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    :type after_scheduler: torch.optim.lr_scheduler
    :param last_epoch: The index of last epoch, defaults to -1
    :type last_epoch: int, optional
    """

    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class WarmupDelayerScheduler(_LRScheduler):
    """ Starts with a linear warmup lr schedule until it reaches N epochs and a flat lr schedule until it reaches M epochs the applies a scheduler 

    :param optimizer: Wrapped optimizer.
    :type optimizer: torch.optim.Optimizer
    :param warmup_epochs: Number of epochs to linearly warmup lr until starting aplying the scheduler
    :type warmup_epochs: int
    :param delay_epochs: Number of epochs to keep the initial lr until starting aplying the scheduler
    :type delay_epochs: int
    :param after_scheduler: After target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    :type after_scheduler: torch.optim.lr_scheduler
    :param last_epoch: The index of last epoch, defaults to -1
    :type last_epoch: int, optional
    """

    def __init__(self, optimizer, warmup_epochs, delay_epochs, after_scheduler, last_epoch=-1):
        if delay_epochs < 0:
            raise ValueError(f'delay_epochs must >= 0, got {delay_epochs}')
        if warmup_epochs < 0:
            raise ValueError(f'warmup_epochs must >= 0, got {warmup_epochs}')
        self.warmup_epochs = warmup_epochs
        self.delay_epochs = delay_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs + self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                # reset lr to base_lr
                for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    group['lr'] = base_lr
                self.finished = True
            with _enable_get_lr_call(self.after_scheduler):
                return self.after_scheduler.get_lr()
        elif self.last_epoch >= self.warmup_epochs:
            return self.base_lrs

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)
