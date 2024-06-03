from typing import List

import torch.distributed as dist
from torch.distributed import ProcessGroup


class ProcessGroupManager:
    """
    ProcessGroupManager is used to manage the process groups in the cluster.

    There are some terms used in this class:
        - pg: the short name for process group
        - pg_name: the name of the process group
        - pg_size: the world size of the process group
        - rank: the rank of the current process in the process group
        - world_size: the total number of processes in the process group
    """

    def __init__(self):
        self.pg_store = dict()

    def create_process_group(self, name: str, ranks: List[int], backend: str = "nccl") -> ProcessGroup:
        """
        Get a process group by name. If the process group does not exist, it will be created.

        Args:
            name (str): name of the process group
            ranks (List[int]): ranks of the process group
            backend (str, optional): backend of the process group. Defaults to 'nccl'.

        Returns:
            ProcessGroup: the process group
        """
        if name not in self.pg_store:
            pg = dist.new_group(ranks=ranks, backend=backend)
            self.pg_store[name] = pg
            return pg
        else:
            raise ValueError(f"Process group {name} already exists.")

    def get(self, name: str) -> ProcessGroup:
        """
        Get a process group by name.

        Args:
            name (str): name of the process group

        Returns:
            ProcessGroup: the process group
        """
        if name in self.pg_store:
            return self.pg_store[name]
        else:
            raise ValueError(f"Process group {name} does not exist.")

    def destroy(self, name: str) -> None:
        """
        Destroy a process group by name.

        Args:
            name (str): name of the process group
        """
        if name in self.pg_store:
            dist.destroy_process_group(self.pg_store[name])
            del self.pg_store[name]
        else:
            raise ValueError(f"Process group {name} does not exist.")

    def destroy_all(self) -> None:
        """
        Destroy all process groups.
        """
        for name in self.pg_store:
            dist.destroy_process_group(self.pg_store[name])
        self.pg_store.clear()
