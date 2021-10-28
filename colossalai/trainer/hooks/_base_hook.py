#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC

from torch import Tensor

from colossalai.logging import get_global_dist_logger
from .._trainer import Trainer


class BaseHook(ABC):
    """This class allows users to add desired actions in specific time points
    during training or evaluation.

    :param trainer: Trainer attached with current hook
    :param priority: Priority in the printing, hooks with small priority will be printed in front
    :type trainer: Trainer
    :type priority: int
    """
    def __init__(self, trainer: Trainer, priority: int) -> None:
        self.trainer = trainer
        self.priority = priority
        self.logger = get_global_dist_logger()

    def before_train(self):
        """Actions before training.
        """
        pass

    def after_train(self):
        """Actions after training.
        """
        pass

    def before_train_iter(self):
        """Actions before running a training iteration.
        """
        pass

    def after_train_iter(self, output: Tensor, label: Tensor, loss: Tensor):
        """Actions after running a training iteration.

        :param output: Output of the model
        :param label: Labels of the input data
        :param loss: Loss between the output and input data
        :type output: Tensor
        :type label: Tensor
        :type loss: Tensor
        """
        pass

    def before_train_epoch(self):
        """Actions before starting a training epoch.
        """
        pass

    def after_train_epoch(self):
        """Actions after finishing a training epoch.
        """
        pass

    def before_test(self):
        """Actions before evaluation.
        """
        pass

    def after_test(self):
        """Actions after evaluation.
        """
        pass

    def before_test_epoch(self):
        """Actions before starting a testing epoch.
        """
        pass

    def after_test_epoch(self):
        """Actions after finishing a testing epoch.
        """
        pass

    def before_test_iter(self):
        """Actions before running a testing iteration.
        """
        pass

    def after_test_iter(self, output: Tensor, label: Tensor, loss: Tensor):
        """Actions after running a testing iteration.

        :param output: Output of the model
        :param label: Labels of the input data
        :param loss: Loss between the output and input data
        :type output: Tensor
        :type label: Tensor
        :type loss: Tensor
        """
        pass

    def init_runner_states(self, key, val):
        """Initializes trainer's state.

        :param key: Key of reseting state
        :param val: Value of reseting state
        """
        if key not in self.trainer.states:
            self.trainer.states[key] = val
