"""Cross-rank lock for TileLang JIT compilation.

TileLang's disk cache only locks inside one process.  When every rank sees a new shape at the same
step, they can all compile the same kernel and dlopen a half-written ``kernel_lib.so`` (observed as
SIGSEGV on four of eight ranks).  The lock must live beside the shared cache, not in each node's
``/tmp``; creating the directory here is deliberate, because a missing ``~/.tilelang`` previously
turned the cross-node lock into a per-node one without any error.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
from typing import Callable

__all__ = ["lock_directory", "locked_compile"]


def lock_directory() -> Path:
    raw = os.environ.get("COSMOS3_TILELANG_LOCK_DIR") or os.environ.get("TILELANG_CACHE_DIR") or "~/.tilelang"
    root = Path(raw).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"cannot create the TileLang compile-lock directory {root}; set COSMOS3_TILELANG_LOCK_DIR "
            "to a shared writable directory"
        ) from exc
    return root


def locked_compile(signature: str, build: Callable[[], object], *, root=None):
    """Run ``build`` while holding a file lock keyed by ``signature``.

    ``root`` is exposed for tests; production callers use ``COSMOS3_TILELANG_LOCK_DIR`` or the
    TileLang cache directory.  The directory is created rather than silently falling back to /tmp.
    """
    directory = Path(root).expanduser() if root is not None else lock_directory()
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(signature).encode()).hexdigest()[:16]
    lock_path = directory / f"cosmos3_tilelang_{digest}.lock"
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            return build()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
