from sqlite3 import SQLITE_CREATE_TRIGGER
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
from typing import List, Optional, Iterator, Tuple
import abc


from files_2022.ColossalAI.colossalai.nn.parallel.layers.cache_embedding.base_embedding import BaseEmbeddingBag

from .freq_aware_embedding import FreqAwareEmbeddingBag

from colossalai.tensor import ColoParameter, ShardSpec, ComputePattern, ProcessGroup, ColoTensorSpec, ColoTensor
from .cache_mgr import CachedParamMgr, EvictionStrategy

class TablewiseEmbeddingBagConfig:
    num_embeddings: int
    cuda_row_num: int
    assigned_rank: int = 0
    buffer_size = 50_000
    ids_freq_mapping = None
    initial_weight: torch.tensor = None # 
    name: str = "" # feature name
'''
example: 
def prepare_tablewise_config(args, cache_ratio, ...):
    embedding_bag_configs: List[TablewiseEmbeddingBagConfig] = []
    ...
    return embedding_bag_configs
'''

def _all_to_all_for_tablewise(x: torch.Tensor, pg: ProcessGroup, scatter_strides: List[int], gather_strides: List[int], forward=True) -> torch.Tensor:
    world_size = pg.tp_world_size()
    rank = pg.tp_local_rank()
    if world_size == 1:
        return x
    assert x.device.type == 'cuda', f"Currently, the collective function dual_all_to_all only supports nccl backend"
    if forward:
        scatter_list = x.split(scatter_strides, 0)
        gather_list = [torch.empty(scatter_strides[rank], gather_strides[i], dtype=x.dtype,
                                   device=x.device) for i in range(world_size)]
        torch.distributed.all_to_all(gather_list, scatter_list, group=pg.tp_process_group)
        return torch.cat(gather_list, 1).contiguous()
    else:
        scatter_list = x.split(scatter_strides, 1)
        gather_list = [torch.empty(gather_strides[i], scatter_strides[rank], dtype=x.dtype,
                                   device=x.device) for i in range(world_size)]
        torch.distributed.all_to_all(gather_list, scatter_list, group=pg.tp_process_group)
        return torch.cat(gather_list, 0).contiguous()

class _DualAllToAllForTablewise(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, pg, scatter_strides, gather_strides):
        ctx.pg = pg
        ctx.scatter_strides = scatter_strides
        ctx.gather_strides = gather_strides
        return _all_to_all_for_tablewise(x, pg, scatter_strides, gather_strides, forward=True)

    @staticmethod
    def backward(ctx, grad):
        return _all_to_all_for_tablewise(grad, ctx.pg, ctx.gather_strides, ctx.scatter_strides, forward=False), None, None, None

def _dual_all_to_all(x, pg, scatter_strides, gather_strides):
    return _DualAllToAllForTablewise.apply(x, pg, scatter_strides, gather_strides)

class ParallelFreqAwareEmbeddingBagTablewise(abc.ABC, nn.Module):
    # I will implement LFU first
    '''
    every table assigned to this FreqAwareEmbeddingBag is managed by a FreqAwareEmbeddingBag.
    '''
    def __init__(self,
                 embedding_bag_configs: List[TablewiseEmbeddingBagConfig],
                 embedding_dim: int,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.,
                 scale_grad_by_freq=False,
                 sparse=False,
                 mode='mean',
                 include_last_offset=False,
                 dtype=None,
                 device=None,
                 warmup_ratio=0.7,
                 pin_weight=False,
                 evict_strategy: EvictionStrategy = EvictionStrategy.LFU):
        super(ParallelFreqAwareEmbeddingBagTablewise, self).__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.global_tables_assign = [config.assigned_rank for config in embedding_bag_configs]
        self.global_tables_length = [config.num_embeddings for config in embedding_bag_configs]
        self.global_tables_num = len(embedding_bag_configs)
        self.global_tables_offsets = torch.cumsum(torch.tensor(self.global_tables_length)) - self.global_tables_length[0]
        self.assigned_tables: List[int] = []
        for i, rank in enumerate(self.global_tables_assign):
            if rank == self.rank:
                self.assigned_tables.append(i)
        self.include_last_offset = include_last_offset
        self.pg = ProcessGroup(tp_degree=self.world_size)
        
        # prepare FreqAwareEmbeddingBag list
        self.freq_aware_embedding_bag_pt: List[FreqAwareEmbeddingBag] = []
        for config in embedding_bag_configs:
            if config.assigned_rank != self.rank:
                continue
            self.freq_aware_embedding_bag_pt.append(
                FreqAwareEmbeddingBag(
                    num_embeddings=config.num_embeddings,
                    embedding_dim=embedding_dim,
                    padding_idx=padding_idx,
                    max_norm=max_norm,
                    norm_type=norm_type,
                    scale_grad_by_freq=scale_grad_by_freq,
                    sparse=sparse,
                    _weight=config.initial_weight,
                    mode=mode,
                    include_last_offset=include_last_offset,
                    dtype=dtype,
                    device=device,
                    cuda_row_num=config.cuda_row_num ,
                    ids_freq_mapping=config.ids_freq_mapping,
                    warmup_ratio=warmup_ratio,
                    buffer_size=config.buffer_size,
                    pin_weight=pin_weight,
                    evict_strategy=evict_strategy
                )
            )

        # prepare list shape for all_to_all output
        self.embedding_dim_per_rank = [0 for i in range(self.world_size)]
        for rank in self.global_tables_assign:
            self.embedding_dim_per_rank[rank] += embedding_dim

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor = None, per_sample_weights=None, shape_hook=None):
        # determine indices to handle
        batch_size = (offsets.shape[0]) // self.num_total_tables
        indices_start_positions = []
        indices_end_positions = []
        for handle_table in self.assigned_tables:
            indices_start_positions.append(offsets[batch_size * handle_table])
            if (not self.include_last_offset) and (batch_size * (handle_table + 1) >= indices.shape[0]):
                indices_end_positions.append(indices.shape[0])
            else:
                indices_end_positions.append(offsets[batch_size * (handle_table + 1)])

        local_output_pt = []
        for i, handle_table in enumerate(self.assigned_tables):
            local_indices = indices[indices_start_positions[i]:indices_end_positions[i]] - \
                self.global_tables_offsets[handle_table]
            if self.include_last_offset:
                local_offsets = offsets[batch_size * handle_table:batch_size
                                        * (handle_table + 1) + 1] - offsets[batch_size * (handle_table)]
            else:
                local_offsets = offsets[batch_size * handle_table:batch_size
                                        * (handle_table + 1)] - offsets[batch_size * (handle_table)]
            local_per_sample_weights = None
            if per_sample_weights != None:
                local_per_sample_weights = per_sample_weights[indices_start_positions[i]:indices_end_positions[i]]
            local_output_pt.append(
                self.freq_aware_embedding_bag_pt[i](
                    local_indices,
                    local_offsets,
                    local_per_sample_weights
                )
            )

        # get result of shape = (batch_size, (len(assigned_tables)*embedding_dim))
        local_output = torch.cat(local_indices, 1)
        # then concatenate those local_output on the second demension.
        # use all_to_all
        remains = batch_size % self.world_size
        scatter_strides = [batch_size // self.world_size + int(i < remains) for i in range(self.world_size)]
        output_full = _dual_all_to_all(local_output, self.pg, scatter_strides, self.embedding_dim_per_rank)
        if shape_hook is not None:
            output_full = shape_hook(output_full)
        return output_full
