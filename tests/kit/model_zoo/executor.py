from typing import Callable, Dict, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer

from colossalai.interface import OptimizerWrapper


def run_fwd(
    model: Module, data: Dict, output_transform_fn: Callable, criterion: Optional[Callable] = None
) -> torch.Tensor:
    """run_fwd
    run fwd for the model

    Args:
        model (torch.nn.Module): a PyTorch model
        data (torch.Tensor): input data
        label (torch.Tensor): label
        criterion (Optional[Callable]): a function of criterion

    Returns:
        torch.Tensor: loss of fwd
    """
    outputs = model(**data)
    outputs = output_transform_fn(outputs)
    if criterion:
        loss = criterion(outputs)
    else:
        loss = next(iter(outputs.values())).sum()
    return loss


def run_fwd_bwd(
    model: Module,
    data: Dict,
    output_transform_fn: Callable,
    criterion: Optional[Callable] = None,
    optimizer: Optional[Union[Optimizer, OptimizerWrapper]] = None,
) -> torch.Tensor:
    """run_fwd_bwd
    run fwd and bwd for the model

    Args:
        model (torch.nn.Module): a PyTorch model
        data (torch.Tensor): input data
        label (torch.Tensor): label
        criterion (Optional[Callable]): a function of criterion

    Returns:
        torch.Tensor: loss of fwd
    """
    loss = run_fwd(model, data, output_transform_fn, criterion)
    if optimizer:
        optimizer.backward(loss)
    else:
        loss.backward()
    return loss
