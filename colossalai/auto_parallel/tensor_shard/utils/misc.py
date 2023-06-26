import functools
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import torch

from colossalai.logging import get_dist_logger
from colossalai.tensor.sharding_spec import ShardingSpec, ShardingSpecException

__all__ = ['ignore_sharding_exception', 'pytree_map']


def ignore_sharding_exception(func):
    """
    A function wrapper to handle the ShardingSpecException in the function.
    If ShardingSpecException occurs, this function will return None.

    Usage:
        # mute the assertion error in the function
        @ignore_sharding_exception
        def do_something():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger = get_dist_logger()
            rst = func(*args, **kwargs)
            return rst
        except ShardingSpecException as e:
            logger.debug(e)
            return None

    return wrapper


def check_sharding_spec_validity(sharding_spec: ShardingSpec, tensor: torch.Tensor):
    """
    This function checks whether the ShardingSpec is valid for the physical tensor.
    This check includes 3 items:
        1. the sharding spec covers all dimensions of the physical tensor
        2. the sharding spec for each dimension is divisible by the number of devices.
        3. the sharding spec's entire shape must match the tensor shape
    #
    """
    # make sure all dims are covered in sharding spec
    sharding_len = len(sharding_spec.sharding_sequence)
    tensor_num_dim = tensor.dim()
    num_devices_in_col = sharding_spec.device_mesh.shape[0]
    num_devices_in_row = sharding_spec.device_mesh.shape[1]
    assert sharding_len == tensor_num_dim, \
        f'The ShardingSpec ({sharding_spec.sharding_sequence}) is created for {sharding_len}-dimension tensor, but the given tensor is {tensor_num_dim}-dimension ({tensor.shape}).'

    # make sure the sharding is valid for each dim
    for i in range(tensor_num_dim):
        dim_size = tensor.shape[i]
        dim_spec = sharding_spec.sharding_sequence[i]

        if str(dim_spec).startswith('S'):
            devices_str = str(dim_spec).lstrip('S')
            num_devices = 1

            if '0' in devices_str:
                num_devices *= num_devices_in_col
            if '1' in devices_str:
                num_devices *= num_devices_in_row

            assert dim_size >= num_devices and dim_size % num_devices == 0, \
                f'The dimension at index {i} has value {dim_size}, but it is sharded over {num_devices} devices.'

    # make sure the entire shape matches the physical tensor shape
    assert sharding_spec.entire_shape == tensor.shape, \
        f'The entire_shape of the sharding spec {sharding_spec.entire_shape} does not match the tensor shape {tensor.shape}'


def pytree_map(obj: Any, fn: Callable, process_types: Union[Type, Tuple[Type]] = (), map_all: bool = False) -> Any:
    """process object recursively, like pytree

    Args:
        obj (:class:`Any`): object to process
        fn (:class:`Callable`): a function to process subobject in obj
        process_types (:class: `type | tuple[type]`): types to determine the type to process
        map_all (:class: `bool`): if map_all is True, then any type of element will use fn

    Returns:
        :class:`Any`: returns have the same structure of `obj` and type in process_types after map of `fn`
    """
    if isinstance(obj, dict):
        return {k: pytree_map(obj[k], fn, process_types, map_all) for k in obj}
    elif isinstance(obj, tuple):
        return tuple(pytree_map(o, fn, process_types, map_all) for o in obj)
    elif isinstance(obj, list):
        return list(pytree_map(o, fn, process_types, map_all) for o in obj)
    elif isinstance(obj, process_types):
        return fn(obj)
    else:
        return fn(obj) if map_all else obj
