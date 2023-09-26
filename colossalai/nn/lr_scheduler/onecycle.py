from torch.optim.lr_scheduler import OneCycleLR as _OneCycleLR


class OneCycleLR(_OneCycleLR):
    r"""Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.
    The 1cycle learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.
    This scheduler is not chainable.
    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

      * A value for total_steps is explicitly provided.
      * A number of epochs (epochs) and a number of steps per epoch (steps_per_epoch) are provided.
        In this case, the number of total steps is inferred by total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.
    The default behaviour of this scheduler follows the fastai implementation of 1cycle, which
    claims that "unpublished work has shown even better results by using only two phases". To
    mimic the behaviour of the original paper instead, set ``three_phase=True``.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        pct_start (float, optional):
            The percentage of the cycle (in number of steps) spent increasing the learning rate, defaults to 0.3.
        anneal_strategy (str, optional): {'cos', 'linear'}, Specifies the annealing strategy:
            "cos" for cosine annealing, "linear" for linear annealing, defaults to 'cos'.
        cycle_momentum (bool, optional): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum', defaults to True.
        base_momentum (float, optional):  Lower momentum boundaries in the cycle for each parameter group.
            Note that momentum is cycled inversely to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr', defaults to 0.85.
        max_momentum (float, optional): Upper momentum boundaries in the cycle for each parameter group.
            Functionally, it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr', defaults to 0.95.
        div_factor (float, optional): Determines the initial learning rate via
            initial_lr = max_lr/div_factor, defaults to 25.0.
        final_div_factor (float, optional): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor, defaults to 10000.0.
        last_epoch (int, optional): The index of the last batch. This parameter is used when resuming a training job.
            Since `step()` should be invoked after each batch instead of after each epoch, this number represents
            the total number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning, defaults to -1

    The ``kwargs`` for initializing torch.optim.lr_scheduler.OneCycleLR should include parameters below:
    ::

        epochs (int, optional, default=None)
        steps_per_epoch (int, optional, default=None)
        three_phase (bool, optional, default=False)
        verbose (bool, optional, default=False)

    More details about kwargs could be found in
    `OneCycleLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR>`_.

    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=10000.0,
        last_epoch=-1,
        **kwargs,
    ):
        max_lrs = list(map(lambda group: group["lr"], optimizer.param_groups))
        super().__init__(
            optimizer,
            max_lrs,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            last_epoch=last_epoch,
        )
