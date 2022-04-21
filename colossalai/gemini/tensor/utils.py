import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d

from colossalai.gemini.tensor.stateful_tensor import StatefulTensorV2


def _convert_tensor(tensor: torch.Tensor) -> StatefulTensorV2:
    if not tensor.is_contiguous():
        raise ValueError('input tensor is not a contiguous Tensor')
    return StatefulTensorV2(tensor)


def convert_parameter(module: torch.nn.Module, param_name: str):
    # Perform some validation first.
    if not hasattr(module, param_name):
        raise ValueError(f'module: {module} does not have parameter with name: {param_name}')

    tensor = getattr(module, param_name)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f'Expected {type(module).__name__}.{param_name} to be a Tensor, but found {type(tensor).__name__}')

    if not tensor.is_contiguous():
        raise ValueError(f'param: {param_name} is not a contiguous Tensor')

    st = _convert_tensor(tensor)

    # Replace param with StatefulTensorV2.

    # Need to delete the attribute first since param_name might be
    # torch.nn.Parameter and can't be replaced with StatefulTensorV2 which is
    # not torch.nn.Parameter.
    delattr(module, param_name)

    # Now we can set the attribute appropriately.
    setattr(module, param_name, st)
