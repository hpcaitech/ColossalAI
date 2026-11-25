"""The default process group owned by a spawned numerical worker: create it, and release it on the way out."""

import functools
from contextlib import contextmanager
from typing import Callable, Iterator

import torch
import torch.distributed as dist

import colossalai


def destroy_worker_process_group():
    """Keep Gloo's Python owner alive until the generic process group is gone.

    PyTorch 2.5's generic ProcessGroup holder destroys its backends while holding
    the GIL. A Gloo worker finishing a Python callback can then deadlock with the
    destructor joining that worker. The Python ProcessGroupGloo holder releases
    the GIL in its destructor, so keep that owner until generic cleanup finishes.
    """
    if not dist.is_initialized():
        return
    backend = None
    if dist.get_backend() == "gloo":
        backend = dist.group.WORLD._get_backend(torch.device("cpu"))
    try:
        dist.destroy_process_group()
    finally:
        # Deliberately release through Gloo's no-GIL Python holder, not as a
        # side effect of the generic ProcessGroup's C++ destructor.
        del backend


@contextmanager
def worker_process_group(rank: int, world_size: int, port: int) -> Iterator[None]:
    """Launch ColossalAI on ``localhost:port`` for a spawned worker and release its group when the block exits.

    A worker that finished passes a barrier first, so no rank tears down the shared transport while a peer still uses
    it. A worker that failed skips it: a failing peer may never reach a collective. Either way the group is released by
    :func:`destroy_worker_process_group`, and an error propagates unchanged.
    """
    try:
        colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
        yield
        dist.barrier()
    finally:
        destroy_worker_process_group()


def worker(fn: Callable) -> Callable:
    """Run a ``spawn`` target ``fn(rank, world_size, port, **kwargs)`` inside :func:`worker_process_group`.

    The wrapper keeps ``fn``'s name, so a module-level worker stays picklable for ``spawn``.
    """

    @functools.wraps(fn)
    def run(rank: int, world_size: int, port: int, **kwargs):
        with worker_process_group(rank, world_size, port):
            return fn(rank, world_size, port, **kwargs)

    return run
