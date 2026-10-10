"""MoE operations, loaded on demand to keep package imports lightweight."""

from importlib import import_module

__all__ = [
    "AllGather",
    "AllToAll",
    "DPGradScalerIn",
    "DPGradScalerOut",
    "EPGradScalerIn",
    "EPGradScalerOut",
    "HierarchicalAllToAll",
    "MoeCombine",
    "MoeDispatch",
    "ReduceScatter",
    "all_to_all_uneven",
    "moe_cumsum",
]


def __getattr__(name):
    if name in __all__:
        operation = getattr(import_module("._operation", __name__), name)
        globals()[name] = operation
        return operation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
