#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import functools
from contextlib import contextmanager

import torch.cuda
from torch import Tensor

from .seed_manager import SeedManager
from ..parallel_mode import ParallelMode

_SEED_MANAGER = SeedManager()


def get_seeds():
    """Returns the seeds of the seed manager.

    Returns:
        dict: The seeds of the seed manager.
    """
    return _SEED_MANAGER.seeds


def get_states(copy=False):
    """Returns the seed states of the seed manager.

    Returns:
        dict: The seed states of the seed manager.
    """
    states = _SEED_MANAGER.seed_states

    if copy:
        new_states = dict()

        for parallel_mode, state in states.items():
            new_states[parallel_mode] = state.clone()
        return new_states
    else:
        return _SEED_MANAGER.seed_states


def get_current_mode():
    """Returns the current mode of the seed manager.

    Returns:
        :class:`torch.ByteTensor`: The current mode of the seed manager.
    """
    return _SEED_MANAGER.current_mode


def add_seed(parallel_mode: ParallelMode, seed: int, overwrite: bool = False):
    """Adds a seed to the seed manager for `parallel_mode`.

    Args:
        parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.
        seed (int): The seed to be added
    Raises:
        AssertionError: Raises an AssertionError if `parallel_mode` is not an instance of
            :class:`colossalai.context.ParallelMode` or the seed for `parallel_mode` has been added.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    _SEED_MANAGER.add_seed(parallel_mode, seed, overwrite)


def set_mode(parallel_mode: ParallelMode):
    """Sets the current mode of the seed manager.

    Args:
        parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    _SEED_MANAGER.set_mode(parallel_mode)


def set_seed_states(parallel_mode: ParallelMode, state: Tensor):
    """Sets the state of the seed manager for `parallel_mode`.

    Args:
        parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.
        state (:class:`torch.Tensor`): the state to be set.

    Raises:
        AssertionError: Raises an AssertionError if `parallel_mode` is not found in the seed manager.
    """
    _SEED_MANAGER.set_state(parallel_mode, state)


def sync_states():
    current_mode = get_current_mode()
    current_states = torch.cuda.get_rng_state()
    set_seed_states(current_mode, current_states)


@contextmanager
def seed(parallel_mode: ParallelMode):
    """ A context for seed switch

    Examples:

        >>> with seed(ParallelMode.DATA):
        >>>     output = F.dropout(input)

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    try:
        # set to new mode
        current_mode = _SEED_MANAGER.current_mode
        yield _SEED_MANAGER.set_mode(parallel_mode)
    finally:
        # recover
        _SEED_MANAGER.set_mode(current_mode)


def with_seed(func, parallel_mode: ParallelMode):
    """
    A function wrapper which executes the function with a specified seed.

    Examples:

        >>> # use with decorator
        >>> @with_seed(ParallelMode.DATA)
        >>> def forward(input):
        >>>     return F.dropout(input)
        >>> out = forward(input)
        >>> # OR use it inline
        >>> def forward(input):
        >>>     return F.dropout(input)
        >>> wrapper_forward = with_seed(forward, ParallelMode.DATA)
        >>> out = wrapped_forward(input)

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # switch mode
        current_mode = _SEED_MANAGER.current_mode
        _SEED_MANAGER.set_mode(parallel_mode)

        # exec func
        out = func(*args, **kwargs)

        # recover state
        _SEED_MANAGER.set_mode(current_mode)

        return out

    return wrapper


def moe_set_seed(seed):
    if torch.cuda.is_available():
        from colossalai.core import global_context as gpc
        global_rank = gpc.get_global_rank()
        diff_seed = seed + global_rank
        add_seed(ParallelMode.TENSOR, diff_seed, True)
        print(f"moe seed condition: {global_rank} with tensor seed {diff_seed}", flush=True)


def reset_seeds():
    _SEED_MANAGER.reset()
