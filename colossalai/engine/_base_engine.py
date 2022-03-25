#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from asyncio.log import logger
from typing import List
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from colossalai.logging import get_dist_logger
from torch import Tensor
from colossalai.engine.ophooks import register_ophooks_recursively, BaseOpHook
from typing import Optional, Type
from colossalai.engine.gradient_handler import BaseGradientHandler
from colossalai.logging import get_dist_logger

class Engine:
    """Basic engine class for training and evaluation. It runs a specific process method
    :meth:`step` which is based on the given :attr:`schedule` over each batch of a dataset.
    It controls a iteration in training.

    Args:
        model (``torch.nn.Module``): The neural network model.
        optimizer (``torch.optim.Optimizer``): Optimizer for updating the parameters.
        criterion (``torch.nn.modules.loss._Loss``, optional): Loss function for calculating loss.
        gradient_handlers (List[``BaseGradientHandler``], optional): A list of gradient handler used in backward.
        clip_grad_norm (float, optional): The norm of gradient clipping.
        ophook_list (list): List of ophook.
        verbose (bool): whether to display log info.

    Examples:
        >>> # define model, criterion, optimizer, lr_scheduler, train_dataloader for your training
        >>> model = ...
        >>> criterion = ...
        >>> optimizer = ...
        >>> train_dataloader = ...
        >>> engine, _, _, _ = colossalai.initialize(model, optimizer, criterion)
        >>> engine.train()
        >>> for inputs, labels in train_dataloader
        >>>     # set gradients to zero
        >>>     engine.zero_grad()
        >>>     # run forward pass
        >>>     outputs = engine(inputs)
        >>>     # compute loss value and run backward pass
        >>>     loss = engine.criterion(outputs, labels)
        >>>     engine.backward(loss)
        >>>     # update parameters
        >>>     engine.step()

    The example of using Engine in training could be find in
    `Training with engine and trainer <https://www.colossalai.org/docs/basics/engine_trainer>`_. and
    `Run resnet cifar10 with engine <https://github.com/hpcaitech/ColossalAI-Examples/blob/main/image/resnet/run_resnet_cifar10_with_engine.py>`_.
    """

    def __init__(self,
                 model: Module,
                 optimizer: Optimizer,
                 criterion: Optional[_Loss] = None,
                 gradient_handlers: Optional[List[BaseGradientHandler]] = None,
                 clip_grad_norm: float = 0.0,
                 ophook_list: Optional[List[BaseOpHook]] = None,
                 verbose: bool = True):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._clip_grad_norm = clip_grad_norm
        self._verbose = verbose
        self._logger = get_dist_logger()

        # state
        self.training = True    # default

        # build gradient handler
        if gradient_handlers:
            self._gradient_handlers = gradient_handlers
        else:
            self._gradient_handlers = []

        if ophook_list is None:
            self._ophook_list = []
        else:
            self._ophook_list = ophook_list
        register_ophooks_recursively(self._model, self._ophook_list)

    @property
    def ophooks(self):
        """show current activated ophooks"""
        return self._ophook_list

    @property
    def model(self):
        """Model attached to the engine"""
        return self._model

    @property
    def optimizer(self):
        """Optimizer attached to the engine"""
        return self._optimizer

    @property
    def criterion(self):
        """Criterion attached to the engine"""
        return self._criterion

    def add_hook(self, ophook: Type[BaseOpHook]) -> None:
        """add necessary hook"""
        # whether this hook exist
        for h in self._ophook_list:
            if type(h) == type(ophook):
                logger = get_dist_logger()
                logger.warning(f"duplicate hooks, at least two instance of {type(ophook)}")
        self._ophook_list.append(ophook)
        register_ophooks_recursively(self._model, self._ophook_list)

    def remove_hook(self, ophook: Type[BaseOpHook]) -> None:
        """remove hook"""
        logger = get_dist_logger()
        logger.warning(f"removing hooks is currently not supported")

    def zero_grad(self):
        """Set the gradient of parameters to zero
        """
        self.optimizer.zero_grad()

    def step(self):
        """Execute parameter update
        """
        self._all_reduce_gradients()
        self.optimizer.clip_grad_norm(self.model, self._clip_grad_norm)
        return self.optimizer.step()

    def backward(self, loss: Tensor):
        """Start backward propagation given the loss value computed by a loss function.

        Args:
            loss (:class:`torch.Tensor`): Loss value computed by a loss function.
        """
        ret = self.optimizer.backward(loss)
        for ophook in self._ophook_list:
            ophook.post_iter()
        return ret

    def backward_by_grad(self, tensor, grad):
        """Start backward propagation given the gradient of the output tensor.

        Args:
            tensor (:class:`torch.Tensor`): Output tensor.
            grad (:class:`torch.Tensor`): Gradient passed back to the output.
        """
        ret = self.optimizer.backward_by_grad(tensor, grad)
        for ophook in self._ophook_list:
            ophook.post_iter()
        return ret

    def __call__(self, *args, **kwargs):
        """Run the forward step for the model.

        Returns:
            Tuple[:class:`torch.Tensor`] or :class:`torch.Tensor`: Output of the model.
        """
        return self.model(*args, **kwargs)

    def _all_reduce_gradients(self):
        """Handles all-reduce operations of gradients across different parallel groups.
        """
        for handler in self._gradient_handlers:
            handler.handle_gradient()

    def train(self):
        """Sets the model to training mode.
        """
        self.training = True
        self._model.train()

    def eval(self):
        """Sets the model to evaluation mode.
        """
        self.training = False
        self._model.eval()