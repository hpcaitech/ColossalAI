import math

import torch


def init_normal(tensor, sigma):
    """Init method based on N(0, sigma)."""
    torch.nn.init.normal_(tensor, mean=0.0, std=sigma)


def output_init_normal(tensor, sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)
    torch.nn.init.normal_(tensor, mean=0.0, std=std)
