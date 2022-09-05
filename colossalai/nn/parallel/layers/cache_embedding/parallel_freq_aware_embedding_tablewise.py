import torch
import torch.distributed as dist
import torch.nn as nn
from typing import List
import abc

from .freq_aware_embedding import FreqAwareEmbeddingBag

from colossalai.tensor import ProcessGroup
from .cache_mgr import EvictionStrategy

from colossalai.nn._ops._utils import dual_all_to_all_tablewise


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


class ParallelFreqAwareEmbeddingBagTablewise(abc.ABC, nn.Module):
    """
    every table assigned to this class instance is managed by a FreqAwareEmbeddingBag.
    """

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
        self.rank_of_tables = [config.assigned_rank for config in embedding_bag_config_list]
        self.global_table_num_embeddings_list = [config.num_embeddings for config in embedding_bag_config_list]
        self.global_tables_num = len(embedding_bag_config_list)
        self.global_tables_offsets = torch.cumsum(torch.tensor([0] + self.global_table_num_embeddings_list), 0).cuda()

        self.assigned_table_list: List[int] = []
        for i, rank in enumerate(self.rank_of_tables):
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
                FreqAwareEmbeddingBag(num_embeddings=config.num_embeddings,
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
                                      cuda_row_num=config.cuda_row_num,
                                      ids_freq_mapping=config.ids_freq_mapping,
                                      warmup_ratio=warmup_ratio,
                                      buffer_size=config.buffer_size,
                                      pin_weight=pin_weight,
                                      evict_strategy=evict_strategy))

        # prepare list shape for all_to_all output
        self.embedding_dim_per_rank = [0 for i in range(self.world_size)]
        for rank in self.rank_of_tables:
            self.embedding_dim_per_rank[rank] += embedding_dim

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor = None, per_sample_weights=None, shape_hook=None):
        # determine indices to handle
        batch_size = (offsets.shape[0]) // self.global_tables_num
        local_output_list = []
        for i, handle_table in enumerate(self.assigned_table_list):
            indices_start_position = offsets[batch_size * handle_table]
            if (not self.include_last_offset) and (batch_size * (handle_table + 1) >= indices.shape[0]):
                # till the end special case
                indices_end_position = indices.shape[0]
            else:
                indices_end_position = offsets[batch_size * (handle_table + 1)]

            local_indices = indices[indices_start_position:indices_end_position] - \
                self.global_tables_offsets[handle_table]
            if self.include_last_offset:
                local_offsets = offsets[batch_size * handle_table:batch_size *
                                        (handle_table + 1) + 1] - offsets[batch_size * (handle_table)]
            else:
                local_offsets = offsets[batch_size * handle_table:batch_size *
                                        (handle_table + 1)] - offsets[batch_size * (handle_table)]
            local_per_sample_weights = None
            if per_sample_weights != None:
                local_per_sample_weights = per_sample_weights[indices_start_position:indices_end_position]
            local_output_list.append(self.freq_aware_embedding_bag_list[i](local_indices, local_offsets,
                                                                           local_per_sample_weights))

        # get result of shape = (batch_size, (len(assigned_table_list)*embedding_dim))
        local_output = torch.cat(local_output_list, 1)
        # then concatenate those local_output on the second demension.
        # use all_to_all
        remains = batch_size % self.world_size
        scatter_strides = [batch_size // self.world_size + int(i < remains) for i in range(self.world_size)]
        output_full = dual_all_to_all_tablewise(local_output, self.pg, scatter_strides, self.embedding_dim_per_rank)
        if shape_hook is not None:
            output_full = shape_hook(output_full)
        return output_full
