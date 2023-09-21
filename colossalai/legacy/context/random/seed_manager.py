#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch import Tensor

from colossalai.legacy.context.parallel_mode import ParallelMode


class SeedManager:
    """This class is a manager of all random seeds involved in the system.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """

    def __init__(self):
        self._current_mode = None
        self._seeds = dict()
        self._seed_states = dict()

    @property
    def current_mode(self):
        return self._current_mode

    @property
    def seeds(self):
        return self._seeds

    @property
    def seed_states(self):
        return self._seed_states

    def set_state(self, parallel_mode: ParallelMode, state: Tensor):
        """Sets the state of the seed manager for `parallel_mode`.

        Args:
            parallel_mode (:class:`colossalai.legacy.context.ParallelMode`): The chosen parallel mode.
            state (:class:`torch.Tensor`): the state to be set.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not found in the seed manager.
        """
        assert parallel_mode in self._seed_states, f"Parallel mode {parallel_mode} is not found in the seed manager"
        self._seed_states[parallel_mode] = state

    def set_mode(self, parallel_mode: ParallelMode):
        """Sets the current mode of the seed manager.

        Args:
            parallel_mode (:class:`colossalai.legacy.context.ParallelMode`): The chosen parallel mode.
        """
        if self.current_mode:
            # save the current state for current mode
            self._seed_states[self._current_mode] = torch.cuda.get_rng_state()

        # set the new state for new mode
        self._current_mode = parallel_mode
        torch.cuda.set_rng_state(self._seed_states[parallel_mode])

    def add_seed(self, parallel_mode: ParallelMode, seed: int, overwrite: bool = False):
        """Adds a seed to the seed manager for `parallel_mode`.

        Args:
            parallel_mode (:class:`colossalai.legacy.context.ParallelMode`): The chosen parallel mode.
            seed (int): The seed to be added.
            overwrite (bool, optional): Whether allows to overwrite the seed that has been set already

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance of :class:`colossalai.legacy.context.ParallelMode`
                or the seed for `parallel_mode` has been added.
        """
        assert isinstance(parallel_mode, ParallelMode), "A valid ParallelMode must be provided"
        if overwrite is False:
            assert parallel_mode not in self._seed_states, f"The seed for {parallel_mode} has been added"
        elif parallel_mode in self._seed_states:
            print(f"Warning: {parallel_mode} seed has been overwritten.", flush=True)

        current_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(seed)
        self._seed_states[parallel_mode] = torch.cuda.get_rng_state()
        self._seeds[parallel_mode] = seed
        torch.cuda.set_rng_state(current_state)

    def reset(self):
        self._current_mode = None
        self._seeds = dict()
        self._seed_states = dict()
