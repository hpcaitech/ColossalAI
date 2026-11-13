"""MoE operations and the expert-parallel layer, loaded on demand to keep package imports lightweight."""

from importlib import import_module

_SUBMODULE = {
    "AllGather": "_operation",
    "AllToAll": "_operation",
    "EPGradScalerIn": "_operation",
    "EPGradScalerOut": "_operation",
    "HierarchicalAllToAll": "_operation",
    "MoeCombine": "_operation",
    "MoeDispatch": "_operation",
    "ReduceScatter": "_operation",
    "all_to_all_uneven": "_operation",
    "moe_cumsum": "_operation",
    "DispatchOutput": "dispatcher",
    "Dispatcher": "dispatcher",
    "NCCLAllToAllDispatcher": "dispatcher",
    "Experts": "experts",
    "GroupedMMExperts": "experts",
    "ReferenceExperts": "experts",
    "EPMoELayer": "layer",
}
__all__ = list(_SUBMODULE)


def __getattr__(name):
    if name in _SUBMODULE:
        value = getattr(import_module(f".{_SUBMODULE[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
