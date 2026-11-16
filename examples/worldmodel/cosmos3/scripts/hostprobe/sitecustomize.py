"""Attribute the training step's host-side hotspots to Python call sites, without touching the trainer.

``operator_details.csv`` from the step profile says one step spends 7.2 s of host time in ``aten::copy_`` and 1.9 s
in ``aten::_local_scalar_dense`` (``.item()``), but the CANN export carries no call stacks, so the profile cannot say
where they come from. This module is installed by putting its directory on ``PYTHONPATH``: CPython imports
``sitecustomize`` at startup, and when the process is the trainer it wraps the suspicious Python-level entry points
and attributes every call to the first frame outside ``torch`` -- the same trick ``probe_transpose_source.py`` uses.

    PYTHONPATH=/mnt/disk0/jiangzs/nzf/hostprobe:$PYTHONPATH HOSTPROBE_OUT=/tmp/hostprobe.json \
      bash scripts/bench_online_step.sh hostprobe 32 2 --max-episodes 64 ...

Off unless ``HOSTPROBE_OUT`` is set, and inert in any process that is not ``train_sft``.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import time
import traceback

_OUT = os.environ.get("HOSTPROBE_OUT")

if _OUT and any("train_sft" in arg for arg in sys.argv):
    import torch

    _STATS = {}

    def _record(kind, started, args):
        elapsed = (time.perf_counter() - started) * 1e6
        stack = traceback.extract_stack(limit=16)[:-3]
        site = "?"
        for frame in reversed(stack):
            name = frame.filename
            if "/torch/" in name or "site-packages/torch" in name or "sitecustomize" in name:
                continue
            site = f"{name.rsplit('/examples/', 1)[-1].rsplit('/ColossalAI/', 1)[-1]}:{frame.lineno} in {frame.name}"
            break
        key = (kind, site)
        entry = _STATS.get(key)
        if entry is None:
            entry = _STATS[key] = [0, 0.0]
        entry[0] += 1
        entry[1] += elapsed

    def _wrap(kind, attribute, owner=torch.Tensor):
        original = getattr(owner, attribute)

        def patched(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                _record(kind, started, args)

        setattr(owner, attribute, patched)

    _wrap("item", "item")
    _wrap("copy_", "copy_")
    _wrap("zero_", "zero_")
    _wrap("cpu", "cpu")
    _wrap("tolist", "tolist")

    def _dump():
        rows = [
            {"kind": kind, "site": site, "calls": calls, "us": elapsed}
            for (kind, site), (calls, elapsed) in sorted(_STATS.items(), key=lambda kv: -kv[1][1])
        ]
        # One file per process: the torchrun agent also matches the "train_sft" argument test, does no tensor work,
        # and would otherwise overwrite the ranks' records with an empty list on the way out.
        path = f"{_OUT}.{os.getpid()}.json"
        try:
            with open(path, "w") as handle:
                json.dump(rows, handle, indent=1)
        except OSError:
            pass

    atexit.register(_dump)
