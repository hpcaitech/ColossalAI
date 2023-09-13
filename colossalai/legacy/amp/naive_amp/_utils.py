from typing import List

from torch import Tensor


def has_inf_or_nan(tensor):
    """Check if tensor has inf or nan values.

    Args:
        tensor (:class:`torch.Tensor`): a torch tensor object

    Returns:
        bool: Whether the tensor has inf or nan. True for yes and False for no.
    """
    try:
        # if tensor is half, the .float() incurs an additional deep copy, but it's necessary if
        # Pytorch's .sum() creates a one-element tensor of the same type as tensor
        # (which is true for some recent version of pytorch).
        tensor_sum = float(tensor.float().sum())
        # More efficient version that can be used if .sum() returns a Python scalar
        # tensor_sum = float(tensor.sum())
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if tensor_sum == float('inf') or tensor_sum == -float('inf') or tensor_sum != tensor_sum:
            return True
        return False


def zero_gard_by_list(tensor_list: List[Tensor], set_to_none: bool = True) -> None:
    """Clear the gradient of a list of tensors,

    Note: copied from torch.optim.optimizer.
    """
    for param in tensor_list:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()
