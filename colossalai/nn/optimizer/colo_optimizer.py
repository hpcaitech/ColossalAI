from typing import List, Union, Mapping, Dict, Any

import torch.optim as optim
from torch import Tensor
from colossalai.tensor.colo_tensor import ColoTensor


class ColoOptimizer(optim.Optimizer):

    def __init__(self, named_params: Mapping[str, Union[Tensor, ColoTensor]], optimizer_class, *optimizer_args,
                 **optimizer_kwargs):
        """
        ColoOptimizer collects all tensors in type of ColoTensor and torch.Tensor,
        then use these tensors as ``params`` for optimizers

        Args:
            named_params (Dict[str, Union[Tensor, ShardedTensor]]) : a Dict
                of parameters, where key is the parameter key, value is either
                Tensor or ColoTensor. This usually used in
                conjunction with model.named_parameters(), the same as PyTorch.
            optimizer_class (torch.optim.Optimizer): the Optimizer to use
                locally, i.e. torch.optim.SGD, torch.optim.Adagrad, etc.
            *optimizer_args: the arguments to initialize the optimizer.
            **optimizer_kwargs: the key-word arguments to initialize the optimizer.

        """
        self._optim = optimizer_class([p for n, p in named_params], *optimizer_args, **optimizer_kwargs)
        self.param_groups = self._optim.param_groups
        self.state = self._optim.state

    def zero_grad(self, set_to_none: bool = False):    # type: ignore[override]
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        self._optim.zero_grad(set_to_none)

    def step(self, closure=None):
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        self._optim.step(closure)

    def state_dict(self) -> Dict[str, Any]:
        """
        Returned state and param_groups will contain parameter keys
        instead of parameter indices like torch.optim.Optimizer.
        """
        return self._optim.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        r"""Loads the ColoOptimizer state.

        Args:
            state_dict (dict): ColoOptimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self._optim.load_state_dict(state_dict)

    def add_param_group(self, param_group: Any):
        r"""Add a new param group
        """
        self._optim.add_param_group(param_group)
