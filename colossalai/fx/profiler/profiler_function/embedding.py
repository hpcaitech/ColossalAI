import torch
from typing import Optional
from ..registry import meta_profiler_function


@meta_profiler_function.register(torch.nn.functional.embedding)
def torch_nn_functional_embedding(
    input: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    # F.embedding is a dictionary lookup, so technically it has 0 FLOPs. (https://discuss.pytorch.org/t/correct-way-to-calculate-flops-in-model/67198/6)
    flops = 0
    macs = 0
    return flops, macs
