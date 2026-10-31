"""Trace: the record of one run, ``{name: tensor}`` plus metadata, that the compare and judge layers consume.

A trace is saved as a single ``.safetensors`` file with its metadata as JSON in the file header, so runs made on
different machines (e.g. GPU and NPU) can be compared offline without unpickling anything.
"""

import json
import os
import platform
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from .compare import flatten_tensors

META_KEY = "numerics_trace"


def environment_meta() -> Dict[str, Any]:
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    try:
        import transformers

        meta["transformers"] = transformers.__version__
    except ImportError:
        pass
    return meta


class Trace(Mapping):
    """Read-only mapping of tensor name to tensor, with a ``meta`` dict describing how the run was made.

    Tensors are detached and kept on CPU in their original dtype. Anything accepted by ``judge``, including another
    trace, can be turned into a trace with ``Trace.capture``.
    """

    def __init__(self, tensors: Dict[str, Tensor], meta: Optional[Dict[str, Any]] = None):
        self._tensors = {name: t.detach().cpu() for name, t in tensors.items()}
        self.meta = dict(meta or {})

    @classmethod
    def capture(cls, outputs: Any, **meta) -> "Trace":
        """Flatten ``outputs`` (tensor, nested dicts / lists, ``ModelOutput``) into a trace; ``meta`` is recorded."""
        return cls(flatten_tensors(outputs), {**environment_meta(), **meta})

    def save(self, path: Union[str, os.PathLike]) -> None:
        tensors = {name: t.contiguous().clone() for name, t in self._tensors.items()}
        save_file(tensors, path, metadata={META_KEY: json.dumps(self.meta, default=str)})

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "Trace":
        with safe_open(path, framework="pt") as f:
            header = f.metadata() or {}
            if META_KEY not in header:
                raise ValueError(f"{path} is a safetensors file but not a numerics trace")
            tensors = {name: f.get_tensor(name) for name in f.keys()}
        return cls(tensors, json.loads(header[META_KEY]))

    def __getitem__(self, name: str) -> Tensor:
        return self._tensors[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    def __repr__(self) -> str:
        return f"Trace({len(self)} tensors, meta={self.meta})"
