#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import List
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from colossalai.logging import get_dist_logger
from torch import Tensor
from colossalai.engine.ophooks import register_ophooks_recursively, BaseOpHook
from typing import Optional
from colossalai.engine.gradient_handler import BaseGradientHandler


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
        >>> model = gpc.config.model.pop('type')(**gpc.config.model)
        >>> criterion = getattr(gpc.config, 'loss_fn', None)
        >>> optimizer = gpc.config.optimizer.pop('type')(model.parameters(), **gpc.config.optimizer)
        >>> lr_scheduler = LinearWarmupLR(optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=5)
        >>> train_dataloader = utils.get_dataloader(train_ds,
        >>>                                seed=42,
        >>>                                batch_size=gpc.config.BATCH_SIZE,
        >>>                                pin_memory=True,
        >>>                                shuffle=True,
        >>>                                drop_last=True)
        >>> # Initialize your engine, train_dataloader, test_dataloader, lr_scheduler
        >>> engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
        >>>                                                              optimizer,
        >>>                                                              criterion,
        >>>                                                              train_dataloader=train_dataloader,
        >>>                                                              lr_scheduler=lr_scheduler)
        >>> # Beginning training progress
        >>> for epoch in range(gpc.config.NUM_EPOCHS):
        >>>     # execute a training iteration
        >>>     engine.train()
        >>>     for img, label in train_dataloader:
        >>>         img = img.cuda()
        >>>         label = label.cuda()
        >>>         # set gradients to zero
        >>>         engine.zero_grad()
        >>>         # run forward pass
        >>>         output = engine(img)
        >>>         # compute loss value and run backward pass
        >>>         train_loss = engine.criterion(output, label)
        >>>         engine.backward(train_loss)
        >>>         # update parameters
        >>>         engine.step()
        >>>     # update learning rate
        >>>     lr_scheduler.step()
        >>>     # execute a testing iteration
        >>>     engine.eval()
        >>>     correct = 0
        >>>     total = 0
        >>>     for img, label in test_dataloader:
        >>>         img = img.cuda()
        >>>         label = label.cuda()
        >>>         # run prediction without back-propagation
        >>>         with torch.no_grad():
        >>>             output = engine(img)
        >>>             test_loss = engine.criterion(output, label)
        >>>         # compute the number of correct prediction
        >>>         pred = torch.argmax(output, dim=-1)
        >>>         correct += torch.sum(pred == label)
        >>>         total += img.size(0)

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

    def calc_loss(self, *args, **kwargs):
        """Compute the loss value.

        Args:
            args (list): Args used in criterion function.
            kwargs (dict): Kwargs used in criterion function.

        Returns:
            :class:`torch.Tensor`: The loss value.
        """
        return self.criterion(*args, **kwargs)

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
