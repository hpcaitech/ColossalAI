import os
import time
from contextlib import contextmanager

import torch
import torch.nn as nn


class SeedManager:
    """
    This class is a random state manager to change random state for different random seed.

    """

    def __init__(self):
        original_state = torch.cuda.get_rng_state()
        seed = int(f"{int(time.time())}{os.environ['RANK']}")
        torch.cuda.manual_seed(int(seed))
        self.dropout_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(original_state)

    def set_mode(self, rng_state):
        torch.cuda.set_rng_state(rng_state)

    def get_current_mode(self):
        current_state = torch.cuda.get_rng_state()
        return current_state

    @contextmanager
    def dropout_mode(self):
        """
        This is a context manager to change the dropout state and recover the original state.

        Usage:
        ::
            >>> with _seed_manager.dropout_mode():
            >>>     input = super().forward(input)
        """
        try:
            current_mode = self.get_current_mode()
            yield self.set_mode(self.dropout_state)
        finally:
            self.dropout_state = self.get_current_mode()
            self.set_mode(current_mode)


_seed_manager = SeedManager()


class Dropout1D(nn.Dropout):

    def __init__(self, p=0.5, inplace=False):
        super().__init__(p, inplace)

    def forward(self, input):
        with _seed_manager.dropout_mode():
            input = super().forward(input)
        return input
