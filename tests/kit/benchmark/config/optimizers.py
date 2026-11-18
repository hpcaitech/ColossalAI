"""The optimizers a model is measured with, by name: ``OPTIMIZERS[name](params, lr=...)`` returns the optimizer.

A plugin that accepts only its own optimizers gets a translated one (``tests.kit.numerics.capture.plugins.
make_optimizer``): GeminiPlugin rebuilds an AdamW as ColossalAI's ``HybridAdam`` and refuses any other optimizer.

A plugin that keeps the optimizer on the CPU (``PluginConfig.cpu_optimizer``: ZeRO with ``cpu_offload``) gets AdamW as
``HybridAdam`` too (``for_cpu``), as ColossalAI's users run it: its CPU Adam kernel, not PyTorch's generic AdamW,
which spends most of the step on the CPU and would make the plugin look far slower than it is.

A new optimizer is added here, under the name ``--optimizer`` and ``--compare optimizer=...`` take, as a
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


def for_cpu(name: str) -> str:
    """The optimizer a plugin that runs it on the CPU uses for ``name``: AdamW as ColossalAI's ``HybridAdam``."""
    return HYBRID_ADAM if name == DEFAULT_OPTIMIZER else name


def hybrid_adam(params: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer:
    """ColossalAI's ``HybridAdam`` with torch.optim.AdamW's defaults: decoupled weight decay 0.01."""
    from colossalai.nn.optimizer import HybridAdam

    return HybridAdam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, adamw_mode=True)


HYBRID_ADAM = "hybrid_adam"
OPTIMIZERS[HYBRID_ADAM] = hybrid_adam
