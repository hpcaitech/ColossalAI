"""The optimizers a model is measured with, by name: ``OPTIMIZERS[name](params, lr=...)`` returns the optimizer.

A plugin that accepts only its own optimizers gets a translated one (``tests.kit.numerics.capture.plugins.
make_optimizer``): GeminiPlugin rebuilds an AdamW as ColossalAI's ``HybridAdam`` and refuses any other optimizer.

An optimizer line adds its optimizer here, under the name ``--optimizer`` and ``--compare optimizer=...`` take, as a
function of the parameters and the learning rate; one that needs parameter groups (e.g. Muon on the 2-D hidden
weights and AdamW on the rest) builds them inside that function.
"""

from typing import Callable, Dict, Iterable

import torch
from torch import nn

DEFAULT_OPTIMIZER = "adamw"

OPTIMIZERS: Dict[str, Callable[..., torch.optim.Optimizer]] = {
    DEFAULT_OPTIMIZER: torch.optim.AdamW,
}


def build_optimizer(name: str, params: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer:
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; known: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name](params, lr=lr)
