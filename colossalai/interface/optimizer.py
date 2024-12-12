from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


class OptimizerWrapper:
    """
    A standard interface for optimizers wrapped by the Booster.

    Args:
        optim (Optimizer): The optimizer to be wrapped.
    """

    def __init__(self, optim: Optimizer):
        self.optim = optim

    @property
    def parameters(self):
        params = []

        for group in self.param_groups:
            params += group["params"]
        return params

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def defaults(self):
        return self.optim.defaults

    def add_param_group(self, *args, **kwargs):
        return self.optim.add_param_group(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        Performs a single optimization step.
        """
        return self.optim.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        """
        Clears the gradients of all optimized `torch.Tensor`.
        """
        self.optim.zero_grad(*args, **kwargs)

    def backward(self, loss: Tensor, inputs=None, retain_graph=False, **kwargs):
        """
        Performs a backward pass on the loss.
        """
        loss.backward(inputs=inputs, retain_graph=retain_graph, **kwargs)

    def backward_by_grad(self, tensor: Tensor, grad: Tensor, inputs: Tensor = None, retain_graph: bool = False):
        """
        Performs a backward pass for dx or dw,
        for dx, we only calculate dx = w*dy here
        for dw, we only calculate dw = x*dy here

        Args:
            tensor (Tensor): y or loss of current chunk;
            grad_tensors (Tensor): dy of current chunk;
            input_obj (Tensor): for dx, input_obj is x of current chunk;
                                for dw, input_obj is w of current chunk;
            retain_graph (bool): default to be True, we retain graph in backward_b
        """
        torch.autograd.backward(
            tensors=tensor,
            grad_tensors=grad,
            inputs=inputs,
            retain_graph=retain_graph,
        )

    def state_dict(self):
        """
        Returns the optimizer state.
        """
        return self.optim.state_dict()

    def load_state_dict(self, *args, **kwargs):
        """
        Loads the optimizer state.
        """
        self.optim.load_state_dict(*args, **kwargs)

    def clip_grad_by_value(self, clip_value: float, *args, **kwargs) -> None:
        """
        Clips gradient of an iterable of parameters at specified min and max values.

        Args:
            clip_value (float or int): maximum allowed value of the gradients. Gradients are clipped in the range

        Note:
            In PyTorch Torch 2.0 and above, you can pass in foreach=True as kwargs to clip_grad_value_ to use the
            faster implementation. Please refer to the PyTorch documentation for more details.
        """
        nn.utils.clip_grad_value_(self.parameters, clip_value, *args, **kwargs)

    def clip_grad_by_norm(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = False,
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Clips gradient norm of an iterable of parameters.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
            error_if_nonfinite (bool): if True, an error is raised if the total norm is non-finite. Default: False

        Note:
            In PyTorch Torch 2.0 and above, you can pass in foreach=True as kwargs to clip_grad_norm_ to use the
            faster implementation. Please refer to the PyTorch documentation for more details.
        """
        norm = nn.utils.clip_grad_norm_(self.parameters, max_norm, norm_type, error_if_nonfinite, *args, **kwargs)
        return norm

    def scale_loss(self, loss: Tensor):
        """
        Scales the loss for mixed precision training.

        Note: Only available for optimizers with mixed precision training.

        Args:
            loss (Tensor): The loss to be scaled.
        """
        raise NotImplementedError(
            "The method scale_loss is only available for optimizers with mixed precision training"
        )

    def unscale_grad(self):
        """
        Unscale the gradients for mixed precision training.

        Note: Only available for optimizers with mixed precision training.
        """
        raise NotImplementedError(
            "The method unscale_grad is only available for optimizers with mixed precision training"
        )

    def unwrap(self):
        """
        Unwrap the optimizer for checkpoint saving/loading.
        """
        return self.optim

    def get_grad_norm(self, norm_type: Union[float, int] = 2.0, **kwargs) -> Optional[float]:
        """
        Returns the gradient norm of an iterable of parameters. This method should be called after optimizer.step().

        Args:
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.

        Returns:
            Optional[float]: Total norm of the gradients (viewed as a single vector). If there are no valid gradients, returns None.
        """
        raise NotImplementedError("The method get_grad_norm is not implemented yet.")


class DistributedOptim(Optimizer):
    def setup_distributed(
        self,
        tp_group: Optional[dist.ProcessGroup] = None,
        dp_group: Optional[dist.ProcessGroup] = None,
        shard_to_working_param: Optional[Dict] = {},
        padding_map: Optional[Dict] = None,
        is_zero: Optional[bool] = False,
    ):
        """Assign process groups for TP and ZeRO 2.
        Arguments:
            tp_group (dist.ProcessGroup): Tensor Parallel process group
            dp_group (dist.ProcessGroup): ZeRO stage 2 process group
            shard_to_working_param (Dict): ZeRO stage 2 feeds the optimizer a sharded param view to match grad shape.
                This maps from id(view) to model params used in forward & backward.
            padding_map (Dict): Per-param padding from ZeRO stage 2
            is_zero (bool): Whether to use ZeRO stage 2.
        """

        raise NotImplementedError("setup_distributed for TP/DP isn't supported by this optimizer yet!")
