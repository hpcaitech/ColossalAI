import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
from typing import List, Optional, Iterator, Tuple
import abc

from .freq_aware_embedding import FreqAwareEmbeddingBag

from colossalai.tensor import ColoParameter, ShardSpec, ComputePattern, ProcessGroup, ColoTensorSpec, ColoTensor
from .cache_mgr import CachedParamMgr, EvictionStrategy


class TablewiseEmbeddingBagConfig:
    '''
    example:
    def prepare_tablewise_config(args, cache_ratio, ...):
        embedding_bag_config_list: List[TablewiseEmbeddingBagConfig] = []
        ...
        return embedding_bag_config_list
    '''
    def __init__(self,
                 num_embeddings: int,
                 cuda_row_num: int,
                 assigned_rank: int = 0,
                 buffer_size=50_000,
                 ids_freq_mapping=None,
                 initial_weight: torch.tensor = None,
                 name: str = ""):
        self.num_embeddings = num_embeddings
        self.cuda_row_num = cuda_row_num
        self.assigned_rank = assigned_rank
        self.buffer_size = buffer_size
        self.ids_freq_mapping = ids_freq_mapping
        self.initial_weight = initial_weight
        self.name = name



def _all_to_all_for_tablewise(x: torch.Tensor, pg: ProcessGroup, scatter_strides: List[int], gather_strides: List[int], forward=True) -> torch.Tensor:
    world_size = pg.tp_world_size()
    rank = pg.tp_local_rank()
    if world_size == 1:
        return x
    assert x.device.type == 'cuda', f"Currently, the collective function dual_all_to_all only supports nccl backend"
    if forward:
        scatter_list = list(x.split(scatter_strides, 0))
        gather_list = [torch.empty(scatter_strides[rank], gather_strides[i], dtype=x.dtype,
                                   device=x.device) for i in range(world_size)]
        torch.distributed.all_to_all(gather_list, scatter_list, group=pg.tp_process_group())
        return torch.cat(gather_list, 1).contiguous()
    else:
        # split on dim 1, lose contiguity
        scatter_list = [each.contiguous() for each in x.split(scatter_strides, 1)]
        gather_list = [torch.empty(gather_strides[i], scatter_strides[rank], dtype=x.dtype,
                                   device=x.device) for i in range(world_size)]
        torch.distributed.all_to_all(gather_list, scatter_list, group=pg.tp_process_group())
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
    '''
    every table assigned to this class instance is managed by a FreqAwareEmbeddingBag.
    '''

    def __init__(self,
                 embedding_bag_config_list: List[TablewiseEmbeddingBagConfig],
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
        self.global_table_assign_list = [config.assigned_rank for config in embedding_bag_config_list]
        self.global_table_num_embeddings_list = [config.num_embeddings for config in embedding_bag_config_list]
        self.global_tables_num = len(embedding_bag_config_list)
        self.global_tables_offsets = torch.cumsum(torch.tensor([0] + self.global_table_num_embeddings_list), 0)

        self.assigned_table_list: List[int] = []
        for i, rank in enumerate(self.global_table_assign_list):
            if rank == self.rank:
                self.assigned_table_list.append(i)
        self.include_last_offset = include_last_offset
        self.pg = ProcessGroup(tp_degree=self.world_size)

        # prepare FreqAwareEmbeddingBag list

        self.freq_aware_embedding_bag_list: nn.ModuleList = nn.ModuleList()
        for config in embedding_bag_config_list:
            if config.assigned_rank != self.rank:
                continue
            self.freq_aware_embedding_bag_list.append(
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
        for rank in self.global_table_assign_list:
            self.embedding_dim_per_rank[rank] += embedding_dim

        #print("global_table_assign_list {}".format(self.global_table_assign_list))
        #print("global_table_num_embeddings_list {}".format(self.global_table_num_embeddings_list))
        #print("global_tables_offsets {}".format(self.global_tables_offsets))
#
    def forward(self, indices: torch.Tensor, offsets: torch.Tensor = None, per_sample_weights=None, shape_hook=None):
        # determine indices to handle
        batch_size = (offsets.shape[0]) // self.global_tables_num
        local_output_list = []
        for i, handle_table in enumerate(self.assigned_table_list):
            indices_start_position = offsets[batch_size * handle_table]
            if (not self.include_last_offset) and (batch_size * (handle_table + 1) >= indices.shape[0]):
                # till the end special case
                indices_end_position = indices.shape[0]
            else :
                indices_end_position = offsets[batch_size * (handle_table + 1)]
            
            local_indices = indices[indices_start_position:indices_end_position] - \
                self.global_tables_offsets[handle_table]
            if self.include_last_offset:
                local_offsets = offsets[batch_size * handle_table:batch_size
                                        * (handle_table + 1) + 1] - offsets[batch_size * (handle_table)]
            else:
                local_offsets = offsets[batch_size * handle_table:batch_size
                                        * (handle_table + 1)] - offsets[batch_size * (handle_table)]
            local_per_sample_weights = None
            if per_sample_weights != None:
                local_per_sample_weights = per_sample_weights[indices_start_position:indices_end_position]
            local_output_list.append(
                self.freq_aware_embedding_bag_list[i](
                    local_indices,
                    local_offsets,
                    local_per_sample_weights
                )
            )

        # get result of shape = (batch_size, (len(assigned_table_list)*embedding_dim))
        local_output = torch.cat(local_output_list, 1)
        # then concatenate those local_output on the second demension.
        # use all_to_all
        remains = batch_size % self.world_size
        scatter_strides = [batch_size // self.world_size + int(i < remains) for i in range(self.world_size)]
        output_full = _dual_all_to_all(local_output, self.pg, scatter_strides, self.embedding_dim_per_rank)
        if shape_hook is not None:
            output_full = shape_hook(output_full)
        return output_full
