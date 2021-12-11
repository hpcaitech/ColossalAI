#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC

from torch import Tensor


class BaseHook(ABC):
    """This class allows users to add desired actions in specific time points
    during training or evaluation.

    :param trainer: Trainer attached with current hook
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type trainer: Trainer
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

        :param output: Output of the model
        :param label: Labels of the input data
        :param loss: Loss between the output and input data
        :type output: Tensor
        :type label: Tensor
        :type loss: Tensor
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

        :param output: Output of the model
        :param label: Labels of the input data
        :param loss: Loss between the output and input data
        :type output: Tensor
        :type label: Tensor
        :type loss: Tensor
        """
        pass

    def init_runner_states(self, trainer, key, val):
        """Initializes trainer's state.

        :param key: Key of reseting state
        :param val: Value of reseting state
        """
        if key not in trainer.states:
            trainer.states[key] = val
