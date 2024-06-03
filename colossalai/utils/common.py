#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import functools
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from colossalai.accelerator import get_accelerator


def get_current_device():
    """
    A wrapper function for accelerator's API for backward compatibility.
    """
    return get_accelerator().get_current_device()


def ensure_path_exists(filename: str):
    # ensure the path exists
    dirpath = os.path.dirname(filename)
    if not os.path.exists(dirpath):
        Path(dirpath).mkdir(parents=True, exist_ok=True)


@contextmanager
def conditional_context(context_manager, enable=True):
    if enable:
        with context_manager:
            yield
    else:
        yield


def is_ddp_ignored(p):
    return getattr(p, "_ddp_to_ignore", False)


def disposable(func: Callable) -> Callable:
    executed = False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal executed
        if not executed:
            executed = True
            return func(*args, **kwargs)

    return wrapper


def free_storage(data: torch.Tensor) -> None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        # Since we're modifying the Tensor's Storage directly, make sure the Tensor
        # is the sole occupant of the Storage.
        assert data.storage_offset() == 0
        data.storage().resize_(0)


def _cast_float(args, dtype: torch.dtype):
    if isinstance(args, torch.Tensor) and torch.is_floating_point(args):
        args = args.to(dtype)
    elif isinstance(args, (list, tuple)):
        args = type(args)(_cast_float(t, dtype) for t in args)
    elif isinstance(args, dict):
        args = {k: _cast_float(v, dtype) for k, v in args.items()}
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
