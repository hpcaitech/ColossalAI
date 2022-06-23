import torch
from colossalai.context.singleton_meta import SingletonMeta
from colossalai.context.parallel_mode import ParallelMode
import numpy as np
import random
import os


class DistributedManager(metaclass=SingletonMeta):
    """DistributedManager 
    A Global Context Manager for ColoTensor
    """

    def __init__(self):
        self.process_groups = dict()
        self.cpu_process_groups = dict()
        self.ranks_in_group = dict()

    def init_default_process_group(self, rank, world_size, host, port, backend):
        init_method = f"tcp://{host}:{port}"
        torch.distributed.init_process_group(rank=rank, world_size=world_size, backend=backend, init_method=init_method)
        group = torch.distributed.GroupMember.WORLD

        ranks = list(range(world_size))
        cpu_group = torch.distributed.new_group(ranks, backend='gloo') \
            if backend != 'gloo' else torch.distributed.GroupMember.WORLD

        self.add_process_group(ParallelMode.GLOBAL, group, cpu_group, ranks)

    def add_process_group(self, name, group, cpu_group, ranks):
        assert name not in self.process_groups, \
            f"Process group name: {name} is already in use"
        self.process_groups[name] = group
        self.cpu_process_groups[name] = cpu_group
        self.ranks_in_group[name] = ranks

    def get_world_size(self, name=ParallelMode.GLOBAL):
        return torch.distributed.get_world_size(self.process_groups[name])

    def get_rank(self, name=ParallelMode.GLOBAL):
        return torch.distributed.get_rank(self.process_groups[name])

    def get_group(self, name=ParallelMode.GLOBAL):
        return self.process_groups[name]

    def get_cpu_group(self, name=ParallelMode.GLOBAL):
        return self.cpu_process_groups[name]

    def get_ranks_in_group(self, name=ParallelMode.GLOBAL):
        return self.ranks_in_group[name]

    def set_device(self, device_ordinal=None):
        if torch.cuda.is_available():
            global_rank = self.get_rank()
            if device_ordinal is None:
                device_ordinal = global_rank % torch.cuda.device_count()
            torch.cuda.set_device(device_ordinal)

    def set_seed(self, seed):
        """
        To achieve reproducible results, it's necessary to fix random seeds
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


DISTMGR = DistributedManager()


def colo_launch(rank, world_size, host, port, backend, local_rank=None, seed=47):
    DISTMGR.init_default_process_group(rank, world_size, host, port, backend)
    DISTMGR.set_device(local_rank)
    DISTMGR.set_seed(seed)


def colo_launch_from_torch(backend='nccl'):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])
    colo_launch(rank=rank, local_rank=local_rank, world_size=world_size, host=host, port=port, backend=backend)
