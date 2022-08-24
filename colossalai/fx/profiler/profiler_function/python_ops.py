import operator
from typing import Any, Tuple
import torch
from ..registry import meta_profiler_function
from colossalai.fx.proxy import ColoProxy


@meta_profiler_function.register(operator.getitem)
def operator_getitem(a: Any, b: Any) -> Tuple[int, int]:
    flops = 0
    macs = 0
    return flops, macs
