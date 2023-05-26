import os
import time

import torch
import torch.nn as nn


class SeedManager:

    def __init__(self):
        self.original_state = torch.cuda.get_rng_state()
        seed = int(f"{int(time.time())}{os.environ['RANK']}")
        print(seed)
        torch.cuda.manual_seed(int(seed))
        self.dropout_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.original_state)

    def dropout_mode(self):
        self.original_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.dropout_state)

    def origin_mode(self):
        self.dropout_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.original_state)


_seed_manager = SeedManager()


class Dropout1D(nn.Dropout):

    def __init__(self, p=0.5, inplace=False):
        super().__init__(p, inplace)

    def forward(self, input):
        _seed_manager.dropout_mode()
        input = super().forward(input)
        _seed_manager.origin_mode()
        return input
