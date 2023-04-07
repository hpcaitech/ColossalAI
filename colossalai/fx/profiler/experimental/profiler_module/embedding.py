from typing import Tuple

import torch

from ..registry import meta_profiler_module


@meta_profiler_module.register(torch.nn.Embedding)
def torch_nn_embedding(self: torch.nn.Embedding, input: torch.Tensor) -> Tuple[int, int]:
    # nn.Embedding is a dictionary lookup, so technically it has 0 FLOPs. (https://discuss.pytorch.org/t/correct-way-to-calculate-flops-in-model/67198/6)
    flops = 0
    macs = 0
    return flops, macs
