import operator
from typing import Any, Tuple

from ..registry import meta_profiler_function


@meta_profiler_function.register(operator.getitem)
def operator_getitem(a: Any, b: Any) -> Tuple[int, int]:
    flops = 0
    macs = 0
    return flops, macs


@meta_profiler_function.register(getattr)
def python_getattr(a: Any, b: Any) -> Tuple[int, int]:
    flops = 0
    macs = 0
    return flops, macs
