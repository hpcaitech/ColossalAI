import functools
import os
from contextlib import contextmanager

import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.context.singleton_meta import SingletonMeta


class DistCoordinator(metaclass=SingletonMeta):
    """
    This class is used to coordinate distributed training. It is a singleton class, which means that there is only one instance of this
    class in the whole program.

    There are some terms that are used in this class:
        - rank: the rank of the current process
        - world size: the total number of processes
        - local rank: the rank of the current process on the current node
        - master: the process with rank 0
        - node master: the process with local rank 0 on the current node


    ```python
    from colossalai.cluster.dist_coordinator import DistCoordinator
    coordinator = DistCoordinator()

    if coordinator.is_master():
        do_something()

    coordinator.print_on_master('hello world')
    ```

    Attributes:
        rank (int): the rank of the current process
        world_size (int): the total number of processes
        local_rank (int): the rank of the current process on the current node
    """

    def __init__(self):
        assert (
            dist.is_initialized()
        ), "Distributed is not initialized. Please call `torch.distributed.init_process_group` or `colossalai.launch` first."
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        # this is often passed by launchers such as torchrun
        self._local_rank = int(os.environ.get("LOCAL_RANK", -1))

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def local_rank(self) -> int:
        return self._local_rank

    def _assert_local_rank_set(self):
        """
        Assert that the local rank is set. This is often passed by launchers such as torchrun.
        """
        assert (
            self.local_rank >= 0
        ), "The environment variable LOCAL_RANK is not set, thus the coordinator is not aware of the local rank of the current process."

    def is_master(self, process_group: ProcessGroup = None) -> bool:
        """
        Check if the current process is the master process (rank is 0). It can accept a sub process group to check the rank 0 with respect to the process.

        Args:
            process_group (ProcessGroup, optional): process group to use for the rank 0 check. Defaults to None, which refers to the default process group.

        Returns:
            bool: True if the current process is the master process, False otherwise
        """
        rank = dist.get_rank(group=process_group)
        return rank == 0

    def is_node_master(self) -> bool:
        """
        Check if the current process is the master process on the current node (local rank is 0).

        Returns:
            bool: True if the current process is the master process on the current node, False otherwise
        """
        self._assert_local_rank_set()
        return self.local_rank == 0

    def is_last_process(self, process_group: ProcessGroup = None) -> bool:
        """
        Check if the current process is the last process (rank is world size - 1). It can accept a sub process group to check the last rank with respect to the process.

        Args:
            process_group (ProcessGroup, optional): process group to use for the last rank check. Defaults to None, which refers to the default process group.

        Returns:
            bool: True if the current process is the last process, False otherwise
        """
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
        return rank == world_size - 1

    def print_on_master(self, msg: str, process_group: ProcessGroup = None):
        """
        Print message only from rank 0.

        Args:
            msg (str): message to print
            process_group (ProcessGroup, optional): process group to use for the rank 0 check. Defaults to None, which refers to the default process group.
        """
        rank = dist.get_rank(group=process_group)
        if rank == 0:
            print(msg)

    def print_on_node_master(self, msg: str):
        """
        Print message only from local rank 0. Local rank 0 refers to the 0th process running the current node.

        Args:
            msg (str): message to print
        """
        self._assert_local_rank_set()
        if self.local_rank == 0:
            print(msg)

    @contextmanager
    def priority_execution(self, executor_rank: int = 0, process_group: ProcessGroup = None):
        """
        This context manager is used to allow one process to execute while blocking all
        other processes in the same process group. This is often useful when downloading is required
        as we only want to download in one process to prevent file corruption.


        ```python
        from colossalai.cluster import DistCoordinator
        dist_coordinator = DistCoordinator()
        with dist_coordinator.priority_execution():
            dataset = CIFAR10(root='./data', download=True)
        ```

        Args:
            executor_rank (int): the process rank to execute without blocking, all other processes will be blocked
            process_group (ProcessGroup, optional): process group to use for the executor rank check. Defaults to None, which refers to the default process group.
        """
        rank = dist.get_rank(group=process_group)
        should_block = rank != executor_rank

        if should_block:
            self.block_all(process_group)

        yield

        if not should_block:
            self.block_all(process_group)

    def destroy(self, process_group: ProcessGroup = None):
        """
        Destroy the distributed process group.

        Args:
            process_group (ProcessGroup, optional): process group to destroy. Defaults to None, which refers to the default process group.
        """
        dist.destroy_process_group(process_group)

    def block_all(self, process_group: ProcessGroup = None):
        """
        Block all processes in the process group.

        Args:
            process_group (ProcessGroup, optional): process group to block. Defaults to None, which refers to the default process group.
        """
        dist.barrier(group=process_group)

    def on_master_only(self, process_group: ProcessGroup = None):
        """
        A function wrapper that only executes the wrapped function on the master process (rank 0).

        ```python
        from colossalai.cluster import DistCoordinator
        dist_coordinator = DistCoordinator()

        @dist_coordinator.on_master_only()
        def print_on_master(msg):
            print(msg)
        ```
        """
        is_master = self.is_master(process_group)

        # define an inner function
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if is_master:
                    return func(*args, **kwargs)

            return wrapper

        return decorator
