import torch

from typing import Iterator, Tuple, Union
import torch.nn as nn
from colossalai.tensor.colo_tensor import ColoTensor


# The function is credited to PyTorch Team
def named_params_with_colotensor(
    module: nn.Module,
    prefix: str = '',
    recurse: bool = True,
) -> Iterator[Tuple[str, Union[nn.Parameter, ColoTensor]]]:
    r"""Returns an iterator over module parameters (together with the
    ColoTensor parameters), yielding both the name of the parameter
    as well as the parameter itself. This is typically passed to a
    :class:torchshard._shard.sharded_optim.ShardedOptimizer

    Args:
        prefix (str): prefix to prepend to all parameter names.
        recurse (bool): if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.

    Yields:
        (string, Union[Tensor, ColoTensor]): Tuple containing
            the name and parameter (or ColoTensor parameter)

    Example:

        >>> model = torch.nn.Linear(*linear_size)
        >>> delattr(model.weight)
        >>> setattr(model.weight, ColoTensor(...))
        >>> for name, param in named_params_with_colotensor(model):
        >>>    if name in ['weight']:
        >>>        print(param.size())

    """
    modules = module.named_modules(prefix=prefix) if recurse else [(prefix, module)]

    memo = set()
    for mod_prefix, mod in modules:
        # find all sharded tensor params
        for name, val in vars(mod).items():
            if isinstance(val, ColoTensor) and val not in memo:
                memo.add(val)
                name = mod_prefix + ('.' if mod_prefix else '') + name
                yield name, val

    # find all nn.Parameters
    for name, val in module.named_parameters():
        yield name, val


def _convert_tensor(tensor: torch.Tensor) -> ColoTensor:
    return ColoTensor(tensor)


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

    # Replace param with ColoTensor.

    # Need to delete the attribute first since param_name might be
    # torch.nn.Parameter and can't be replaced with ColoTensor which is
    # not torch.nn.Parameter.
    delattr(module, param_name)

    # Now we can set the attribute appropriately.
    setattr(module, param_name, st)
