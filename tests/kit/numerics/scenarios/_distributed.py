"""Release the default process group owned by a spawned numerical worker."""

import torch
import torch.distributed as dist


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
