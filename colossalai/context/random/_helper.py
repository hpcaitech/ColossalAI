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

    :return: The seeds of the seed manager
    :rtype: dict
    """
    return _SEED_MANAGER.seeds


def get_states(copy=False):
    """Returns the seed states of the seed manager.

    :return: The seed states of the seed manager
    :rtype: dict
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

    :return: The current mode of the seed manager.
    :rtype: :class:`torch.ByteTensor`
    """
    return _SEED_MANAGER.current_mode


def add_seed(parallel_mode: ParallelMode, seed: int, overwrite: bool = False):
    """Adds a seed to the seed manager for `parallel_mode`.

    :param parallel_mode: The chosen parallel mode
    :type parallel_mode: :class:`colossalai.context.ParallelMode`
    :param seed: The seed to be added
    :type seed: int
    :raises AssertionError: Raises an AssertionError if `parallel_mode` is not an instance of
        :class:`colossalai.context.ParallelMode` or the seed for `parallel_mode` has been added
    """
    _SEED_MANAGER.add_seed(parallel_mode, seed, overwrite)


def set_mode(parallel_mode: ParallelMode):
    """Sets the current mode of the seed manager.

    :param parallel_mode: The chosen parallel mode
    :type parallel_mode: :class:`colossalai.context.ParallelMode`
    """
    _SEED_MANAGER.set_mode(parallel_mode)


def set_seed_states(parallel_mode: ParallelMode, state: Tensor):
    """Sets the state of the seed manager for `parallel_mode`.

    :param parallel_mode: The chosen parallel mode
    :type parallel_mode: :class:`colossalai.context.ParallelMode`
    :param state: the state to be set
    :type state: :class:`torch.Tensor`
    :raises AssertionError: Raises an AssertionError if `parallel_mode` is not found in the seed manager
    """
    _SEED_MANAGER.set_state(parallel_mode, state)


def sync_states():
    current_mode = get_current_mode()
    current_states = torch.cuda.get_rng_state()
    set_seed_states(current_mode, current_states)


@contextmanager
def seed(parallel_mode: ParallelMode):
    """ A context for seed switch

    Examples::

        with seed(ParallelMode.DATA):
            output = F.dropout(input)

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

    Examples::

        # use with decorator
        @with_seed(ParallelMode.DATA)
        def forward(input):
            return F.dropout(input)
        out = forward(input)
        # OR use it inline
        def forward(input):
            return F.dropout(input)
        wrapper_forward = with_seed(forward, ParallelMode.DATA)
        out = wrapped_forward(input)

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
        moe_mp_rank = gpc.get_local_rank(ParallelMode.MOE_MODEL)
        moe_mp_seed = seed + moe_mp_rank
        add_seed(ParallelMode.MOE_MODEL, moe_mp_seed)

        global_rank = gpc.get_global_rank()
        add_seed(ParallelMode.TENSOR, global_rank, True)
        print(f"moe seed condition: {global_rank} with moe seed {moe_mp_seed}, ",
              f"tensor seed {global_rank}", flush=True)
