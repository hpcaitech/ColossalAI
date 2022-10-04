from typing import List, Any, Tuple, Dict, Callable, Type, Union

import torch
from torch.futures import Future

from colorama import Back, Style

# config for debug and test
use_color_debug = False


def color_debug(text, prefix=' ', color='blue'):
    color = color.upper()
    print(getattr(Back, color), prefix, Style.RESET_ALL, text)


def pytree_map(obj: Any, fn: Callable, process_types: Union[Type, Tuple[Type]] = (), map_all: bool = False) -> Any:
    """process object recursively, like pytree

    Args:
        obj (:class:`Any`): object to process
        fn (:class:`Callable`): a function to process subobject in obj
        process_types(:class: `type | tuple[type]`): types to determine the type to process

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


def tensor_shape_list(obj):
    return pytree_map(obj, fn=lambda x: x.shape, process_types=torch.Tensor)


def get_batch_lengths(batch):
    lengths = []
    pytree_map(batch, fn=lambda x: lengths.append(len(x)), process_types=torch.Tensor)
    return lengths


def split_batch(batch: Any, start, stop, device: str):
    if device == 'cuda':
        fn = lambda x: x[start:stop].cuda()
    else:
        fn = lambda x: x[start:stop]
    return pytree_map(batch, fn=fn, process_types=torch.Tensor)


def type_detail(obj):
    return pytree_map(obj, lambda x: type(x), map_all=True)


def get_real_args_kwargs(args_or_kwargs):
    args_or_kwargs = pytree_map(args_or_kwargs, fn=lambda x: x.wait(), process_types=Future)
    # TODO : combine producer and consumer
    # by default, merge all args in the output args or kwargs
    if args_or_kwargs is not None:
        if isinstance(args_or_kwargs, dict):
            pass
        else:
            flatten_args = []
            pytree_map(args_or_kwargs, fn=lambda x: flatten_args.append(x), map_all=True)
            args_or_kwargs = flatten_args

    return args_or_kwargs
