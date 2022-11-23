from typing import Any

import torch

from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.optimizer.zero_optimizer import ZeroOptimizer

__all__ = ['GeminiAdamOptimizer']


class GeminiAdamOptimizer(ZeroOptimizer):

    def __init__(self, model: torch.nn.Module, **defaults: Any) -> None:
        optimizer = HybridAdam(model.parameters(), **defaults)
        super().__init__(optimizer, model, **defaults)
