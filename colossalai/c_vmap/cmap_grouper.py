import torch
import torch.distributed as dist

from colossalai.core import global_context as gpc

class cmap_group_init:
    def __init__(self, topology: list):
        flat_top = self.flatten_topology(topology)
        self.cmap_group = dist.new_group(ranks=flat_top)
        self.groups = self.group_topology(topology)


    def flatten_topology(self, topology):
        flat_top = []
        for i in topology:
            if isinstance(i, int):
                flat_top.append(i)
            elif isinstance(i, tuple) or isinstance(i, list):
                for j in i:
                    flat_top.append(j)
            else:
                raise TypeError(f"values in topology must be int or iterable of int")
        return flat_top

    def group_topology(self, topology):
        groups = []
        for i in topology:
            if isinstance(i, int):
                groups.append(dist.new_group(ranks=[i]))
            else:
                groups.append(dist.new_group(ranks=list(i)))
        return groups
    