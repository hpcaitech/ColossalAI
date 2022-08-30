import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
from typing import List, Optional, Iterator, Tuple
import abc


from files_2022.ColossalAI.colossalai.nn.parallel.layers.cache_embedding.base_embedding import BaseEmbeddingBag

from .freq_aware_embedding import FreqAwareEmbeddingBag
from colossalai.nn._ops._utils import dual_all_to_all

from colossalai.tensor import ColoParameter, ShardSpec, ComputePattern, ProcessGroup, ColoTensorSpec, ColoTensor
from .cache_mgr import CachedParamMgr, EvictionStrategy


class ParallelFreqAwareEmbeddingBagTablewise(abc.ABC, nn.Module):
    # I will implement LFU first
    '''
    1. per table, "pt", for short. parameters with _pt end are lists of parameters per table.
    2. every table assigned to this FreqAwareEmbeddingBag is managed by a FreqAwareEmbeddingBag.
    '''

    def __init__(self,
                 tables_assign,
                 table_length_offsets,
                 num_embeddings_pt,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.,
                 scale_grad_by_freq=False,
                 sparse=False,
                 _weight_pt=None,
                 mode='mean',
                 include_last_offset=False,
                 dtype=None,
                 device=None,
                 cuda_row_num_pt=0,
                 ids_freq_mapping_pt=None,
                 warmup_ratio=0.7,
                 buffer_size=50_000,
                 pin_weight=False,
                 evict_strategy: EvictionStrategy = EvictionStrategy.LFU):
        '''
        Args:
        tables_assign (List[int]): what rank a table is assigned to. 
            for example, 3 ranks, 6 tables in total. Table 0,1 are assigned to rank 0, table 3,4 are assigned to rank 1,
            table 2,5 is assigned to rank 2. Then tables_assign should be [0,0,2,1,1,2].
        table_length_offsets (List[long]): idx offsets for each table. 
        '''
        super(ParallelFreqAwareEmbeddingBagTablewise, self).__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.tables_assign = tables_assign
        self.table_length_offsets = table_length_offsets
        self.num_total_tables = len(self.tables_assign)
        self.handle_tables:List[int] = []
        for i, rank in enumerate(self.tables_assign):
            if rank == self.rank:
                self.handle_tables.append(i)
        
        self.include_last_offset = include_last_offset
        if _weight is None:
            colo_tensor_spec = ColoTensorSpec(pg=ProcessGroup(tp_degree=self.world_size),
                                              dist_attr=ShardSpec(dims=[-1], num_partitions=[self.world_size]),
                                              compute_attr=ComputePattern.TP1D)

        self.freq_aware_embedding_bag_pt: List[FreqAwareEmbeddingBag] = []
        for num_embeddings, _weight, cuda_row_num, ids_freq_mapping in \
                zip(num_embeddings_pt, _weight_pt, cuda_row_num_pt, ids_freq_mapping_pt):
            # "override" _weight_alloc by building colotensor _weight.
            if _weight is None:
                weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
                with torch.no_grad():
                    weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
                    if padding_idx is not None:
                        weight[padding_idx].fill_(0)
                _weight = ColoTensor.from_torch_tensor(weight, spec=colo_tensor_spec)

            self.freq_aware_embedding_bag_pt.append(
                FreqAwareEmbeddingBag(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    padding_idx=padding_idx,
                    max_norm=max_norm,
                    norm_type=norm_type,
                    scale_grad_by_freq=scale_grad_by_freq,
                    sparse=sparse,
                    _weight=_weight,
                    mode=mode,
                    include_last_offset=include_last_offset,
                    dtype=dtype,
                    device=device,
                    cuda_row_num=cuda_row_num,
                    ids_freq_mapping=ids_freq_mapping,
                    warmup_ratio=warmup_ratio,
                    buffer_size=buffer_size,
                    pin_weight=pin_weight,
                    evict_strategy=evict_strategy
                )
            )
            
            # prepare for all_gather 
            self.all_gather_shape = 
        
    def forward(self, indices: torch.Tensor, offsets: torch.Tensor =None, per_sample_weights=None, shape_hook=None):
        # determine indices to handle
        batch_size = (offsets.shape[0]) // self.num_total_tables
        indices_start_positions = []
        indices_end_positions = []
        for handle_table in self.handle_tables:
            indices_start_positions.append(offsets[batch_size * handle_table])
            if (not self.include_last_offset) and (batch_size * (handle_table + 1) >= indices.shape[0]):
                indices_end_positions.append(indices.shape[0])
            else:
                indices_end_positions.append(offsets[batch_size * (handle_table + 1)])
                
        local_output_pt = []
        for i,handle_table in enumerate(self.handle_tables):
            local_indices = indices[indices_start_positions[i]:indices_end_positions[i]] - self.table_length_offsets[handle_table]
            if self.include_last_offset:
                local_offsets = offsets[batch_size * handle_table:batch_size * (handle_table+1) + 1] - offsets[batch_size * (handle_table)]
            else:
                local_offsets = offsets[batch_size * handle_table:batch_size * (handle_table+1)] - offsets[batch_size * (handle_table)]
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
            
        # all_gather
        local_output_pt = torch.tensor(local_indices, device=torch.cuda.current_device())

        dist.all_gather()
        
