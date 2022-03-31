#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC

from torch import Tensor


class BaseHook(ABC):
    """This class allows users to add desired actions in specific time points
    during training or evaluation.

    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type priority: int
    """

    def __init__(self, priority: int) -> None:
        self.priority = priority

    def after_hook_is_attached(self, trainer):
        """Actions after hooks are attached to trainer.
        """
        pass

    def before_train(self, trainer):
        """Actions before training.
        """
        pass

    def after_train(self, trainer):
        """Actions after training.
        """
        pass

    def before_train_iter(self, trainer):
        """Actions before running a training iteration.
        """
        pass

    def after_train_iter(self, trainer, output: Tensor, label: Tensor, loss: Tensor):
        """Actions after running a training iteration.

        Args:
           trainer (:class:`Trainer`): Trainer which is using this hook.
           output (:class:`torch.Tensor`): Output of the model.
           label (:class:`torch.Tensor`): Labels of the input data.
           loss (:class:`torch.Tensor`): Loss between the output and input data.
        """
        pass

    def before_train_epoch(self, trainer):
        """Actions before starting a training epoch.
        """
        pass

    def after_train_epoch(self, trainer):
        """Actions after finishing a training epoch.
        """
        pass

    def before_test(self, trainer):
        """Actions before evaluation.
        """
        pass

    def after_test(self, trainer):
        """Actions after evaluation.
        """
        pass

    def before_test_epoch(self, trainer):
        """Actions before starting a testing epoch.
        """
        pass

    def after_test_epoch(self, trainer):
        """Actions after finishing a testing epoch.
        """
        pass

    def before_test_iter(self, trainer):
        """Actions before running a testing iteration.
        """
        pass

    def after_test_iter(self, trainer, output: Tensor, label: Tensor, loss: Tensor):
        """Actions after running a testing iteration.

        Args:
           trainer (:class:`Trainer`): Trainer which is using this hook
           output (:class:`torch.Tensor`): Output of the model
           label (:class:`torch.Tensor`): Labels of the input data
           loss (:class:`torch.Tensor`): Loss between the output and input data
        """
        pass

    def init_runner_states(self, trainer, key, val):
        """Initializes trainer's state.

        Args:
            trainer (:class:`Trainer`): Trainer which is using this hook
            key: Key of state to be reset
            val: Value of state to be reset
        """
        if key not in trainer.states:
            trainer.states[key] = val
