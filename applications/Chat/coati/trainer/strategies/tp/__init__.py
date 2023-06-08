from torch.nn import Module

from colossalai.lazy import LazyTensor

from .policy import POLICY_MAP


def _replace_recursively(model: Module) -> None:
    recurse: bool = True
    if type(model) in POLICY_MAP:
        policy = POLICY_MAP[type(model)]()
        recurse = policy.replace(model)
    if recurse:
        for child in model.children():
            _replace_recursively(child)


def tp_parallelize(model: Module) -> None:
    _replace_recursively(model)
    for p in model.parameters():
        if isinstance(p, LazyTensor):
            p.materialize()
    for buf in model.buffers():
        if isinstance(buf, LazyTensor):
            buf.materialize()


__all__ = ["tp_parallelize"]
