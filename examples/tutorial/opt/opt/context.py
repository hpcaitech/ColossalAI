import torch.distributed as dist

from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc


class barrier_context:
    """
    This context manager is used to allow one process to execute while blocking all
    other processes in the same process group. This is often useful when downloading is required
    as we only want to download in one process to prevent file corruption.
    Args:
        executor_rank (int): the process rank to execute without blocking, all other processes will be blocked
        parallel_mode (ParallelMode): the parallel mode corresponding to a process group
    Usage:
        with barrier_context():
            dataset = CIFAR10(root='./data', download=True)
    """

    def __init__(self, executor_rank: int = 0, parallel_mode: ParallelMode = ParallelMode.GLOBAL):
        # the class name is lowercase by convention
        current_rank = gpc.get_local_rank(parallel_mode=parallel_mode)
        self.should_block = current_rank != executor_rank
        self.group = gpc.get_group(parallel_mode=parallel_mode)

    def __enter__(self):
        if self.should_block:
            dist.barrier(group=self.group)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.should_block:
            dist.barrier(group=self.group)
