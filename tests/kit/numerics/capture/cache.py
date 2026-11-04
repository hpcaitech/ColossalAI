"""Caching of reference runs, for real weights where the float64 truth takes long.

A cached trace is reused only if everything that determines it is unchanged; the key hashes:

- the entry name, seed, number of steps and upstream gradient seed, and the parameters it declares unused or tied;
- the reference weights and buffers and every batch, byte for byte, and the code that selects outputs and routers;
- the optimizer (its class, and the hyperparameters and size of each parameter group, read from one built on the
  reference's parameters, which allocates no state);
- for ``ref``, the precision recipe (the float64 truth does not depend on it, so one truth serves every
  configuration of an entry);
- the torch and transformers versions and the device name;
- the source of the capture code, so that a change to how the reference runs invalidates old entries by itself.

Entries are ``<directory>/<entry>/<role>-<key>.safetensors``; nothing is ever overwritten in place. New directories
and files are group-writable, so that a cache shared by a team stays usable by all of it.
"""

import hashlib
import inspect
import json
import os
from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor, nn

from colossalai.accelerator import get_accelerator

from ..trace import Trace, environment_meta
from . import fwd_bwd, plugins, recipes, training, utils

_SOURCES = (training, utils, recipes, plugins, fwd_bwd)


def reference_key(s: "training.Prepared", optimizer_fn: Callable, grad_seed: int, recipe=None) -> str:
    """The cache key of a reference run of ``s``; ``recipe=None`` for the float64 truth."""
    digest = hashlib.sha256()
    _update(digest, dict(entry=s.name, seed=s.seed, steps=len(s.batches), grad_seed=grad_seed, code=s.code))
    _update(digest, dict(unused=sorted(s.unused_params), tied=sorted(sorted(g) for g in s.tied_params)))
    _update(digest, dict(recipe=repr(recipe), optimizer=_optimizer_signature(optimizer_fn, s.reference)))
    _update(digest, _environment())
    for name, tensor in sorted(s.reference.state_dict().items()):
        _update_tensor(digest, name, tensor)
    for step, batch in enumerate(s.batches):
        for name, value in sorted(batch.items()):
            if isinstance(value, Tensor):
                _update_tensor(digest, f"{step}/{name}", value)
            else:
                _update(digest, {f"{step}/{name}": repr(value)})
    for module in _SOURCES:
        digest.update(inspect.getsource(module).encode())
    return digest.hexdigest()[:32]


def cached_run(directory: Optional[str], name: str, role: str, key: str, run: Callable[[], Trace]) -> Trace:
    """``run()``, or the trace it saved earlier under the same key. Without ``directory``, always runs."""
    if directory is None:
        return run()
    path = os.path.join(directory, name, f"{role}-{key}.safetensors")
    if os.path.exists(path):
        trace = Trace.load(path)
        return Trace(dict(trace.items()), dict(trace.meta, cached=path))
    trace = run()
    _make_shared_dir(os.path.dirname(path))
    partial_path = f"{path}.{os.getpid()}.tmp"
    trace.save(partial_path)
    _share(partial_path, 0o664)
    os.replace(partial_path, path)  # atomic: a concurrent reader sees a whole file or none
    return trace


def _make_shared_dir(directory: str) -> None:
    # a shared cache is written by several users, often as root inside a container: keep it group-writable
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
        _share(directory, 0o2775)


def _share(path: str, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except PermissionError:  # created by someone else in the meantime
        pass


def _optimizer_signature(optimizer_fn: Callable, module: nn.Module) -> Dict[str, Any]:
    # the real parameters: an optimizer_fn may group or leave out parameters by shape
    optimizer = optimizer_fn(list(module.parameters()))
    groups = [
        dict({k: repr(v) for k, v in sorted(g.items()) if k != "params"}, size=len(g["params"]))
        for g in optimizer.param_groups
    ]
    return dict(type=type(optimizer).__qualname__, groups=groups)


def _environment() -> Dict[str, str]:
    meta = environment_meta()
    environment = dict(torch=meta["torch"], transformers=meta.get("transformers"), device=get_accelerator().name)
    if environment["device"] == "cuda":
        environment["device"] = torch.cuda.get_device_name(0)
    return environment


def _update(digest, value: Dict[str, Any]) -> None:
    digest.update(json.dumps(value, sort_keys=True, default=repr).encode())


def _update_tensor(digest, name: str, tensor: Tensor) -> None:
    tensor = tensor.detach().cpu().contiguous()
    _update(digest, dict(name=name, dtype=str(tensor.dtype), shape=list(tensor.shape)))
    digest.update(tensor.view(-1).view(torch.uint8).numpy().tobytes())
