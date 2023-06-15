import torch
import torch.distributed as dist
import torch.nn as nn

from .utils import create_randomizer_with_offset


class Dropout1D(nn.Dropout):

    def __init__(self, p=0.5, inplace=False, process_group=None):
        super().__init__(p, inplace)

        # offset the seed with randomizer index and rank
        seed = torch.random.initial_seed()
        self.randomizer = create_randomizer_with_offset(seed, process_group=process_group)

    def forward(self, input):
        with self.randomizer.fork_rng():
            input = super().forward(input)
        return input
