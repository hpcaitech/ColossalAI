import torch.nn as nn
import torch.distributed as dist
from typing import List, Tuple, Union
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc


class PipelineSharedModuleWrapper:

    def __init__(self, pipeline_ranks: Union[List[int], Tuple[int]]) -> None:
        assert len(pipeline_ranks) > 1, f'Expect len(pipeline_ranks) > 1, got {len(pipeline_ranks)}'
        self.pipeline_ranks = pipeline_ranks
        self.group = None
        self.ranks_in_group = None
        self._init_group()

    def _init_group(self):
        world_size = gpc.get_world_size(ParallelMode.GLOBAL)
        dp_size = gpc.get_world_size(ParallelMode.DATA)
        pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
        rank = gpc.get_global_rank()
        num_dp_groups = world_size // dp_size
        num_pp_stages = num_dp_groups // pp_size
        for i in range(dp_size):
            for j in range(num_pp_stages):
                pipeline_ranks = list(range(i * num_dp_groups + j, (i + 1) * num_dp_groups, num_pp_stages))
                sub_ranks = [pipeline_ranks[idx] for idx in self.pipeline_ranks]
                group = dist.new_group(sub_ranks)
                if rank in sub_ranks:
                    self.group = group
                    self.ranks_in_group = sub_ranks

    def register_module(self, module: nn.Module):
        assert self.ranks_in_group is not None,\
            f'Rank {gpc.get_local_rank(ParallelMode.PIPELINE)} is not in pipeline_ranks {self.pipeline_ranks}'
        src = self.ranks_in_group[self.pipeline_ranks[0]]
        for p in module.parameters():
            setattr(p, 'pipeline_shared_module_pg', self.group)
            dist.broadcast(p, src, group=self.group)

    def register_parameter(self, param: nn.Parameter):
        assert self.ranks_in_group is not None,\
            f'Rank {gpc.get_local_rank(ParallelMode.PIPELINE)} is not in pipeline_ranks {self.pipeline_ranks}'
        src = self.ranks_in_group[self.pipeline_ranks[0]]
        setattr(param, 'pipeline_shared_module_pg', self.group)
        dist.broadcast(param, src, group=self.group)
