import tempfile
from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch.distributed as dist


@contextmanager
def shared_tempdir() -> Iterator[str]:
    """
    A temporary directory that is shared across all processes.
    """
    ctx_fn = tempfile.TemporaryDirectory if dist.get_rank() == 0 else nullcontext
    with ctx_fn() as tempdir:
        try:
            obj = [tempdir]
            dist.broadcast_object_list(obj, src=0)
            tempdir = obj[0]  # use the same directory on all ranks
            yield tempdir
        finally:
            dist.barrier()
