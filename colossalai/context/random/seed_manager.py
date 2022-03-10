#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch import Tensor

from colossalai.context.parallel_mode import ParallelMode


class SeedManager:
    """This class is a manager of all random seeds involved in the system.
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

        :param parallel_mode: The chosen parallel mode
        :type parallel_mode: :class:`colossalai.context.ParallelMode`
        :param state: the state to be set
        :type state: :class:`torch.Tensor`
        :raises AssertionError: Raises an AssertionError if `parallel_mode` is not found in the seed manager
        """
        assert parallel_mode in self._seed_states, f'Parallel mode {parallel_mode} is not found in the seed manager'
        self._seed_states[parallel_mode] = state

    def set_mode(self, parallel_mode: ParallelMode):
        """Sets the current mode of the seed manager.

        :param parallel_mode: The chosen parallel mode
        :type parallel_mode: :class:`colossalai.context.ParallelMode`
        """
        if self.current_mode:
            # save the current state for current mode
            self._seed_states[self._current_mode] = torch.cuda.get_rng_state()

        # set the new state for new mode
        self._current_mode = parallel_mode
        torch.cuda.set_rng_state(self._seed_states[parallel_mode])

    def add_seed(self, parallel_mode: ParallelMode, seed: int, overwrtie: bool = False):
        """Adds a seed to the seed manager for `parallel_mode`.

        :param parallel_mode: The chosen parallel mode
        :type parallel_mode: :class:`colossalai.context.ParallelMode`
        :param seed: The seed to be added
        :type seed: int
        :param overwrtie: Whether allows to overwrite the seed that has been set already
        :type overwrtie: bool, optional
        :raises AssertionError: Raises an AssertionError if `parallel_mode` is not an instance of
            :class:`colossalai.context.ParallelMode` or the seed for `parallel_mode` has been added
        """
        assert isinstance(
            parallel_mode, ParallelMode), 'A valid ParallelMode must be provided'
        if overwrtie is False:
            assert parallel_mode not in self._seed_states, f'The seed for {parallel_mode} has been added'
        elif parallel_mode in self._seed_states:
            print(f"Warnning: {parallel_mode} seed has been overwritten.", flush=True)

        current_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(seed)
        self._seed_states[parallel_mode] = torch.cuda.get_rng_state()
        self._seeds[parallel_mode] = seed
        torch.cuda.set_rng_state(current_state)
