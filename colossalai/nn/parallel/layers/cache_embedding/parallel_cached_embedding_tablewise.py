import torch
import torch.distributed as dist
import torch.nn.functional as F

from .cached_embedding import CachedEmbeddingBag
from .cache_mgr import EvictionStrategy
from .embedding_config import TablewiseEmbeddingBagConfig
from colossalai.tensor import ProcessGroup
from colossalai.nn._ops._utils import dual_all_to_all_tablewise

from typing import List
import time


class ParallelCachedEmbeddingBagTablewise(CachedEmbeddingBag):
    """
    all tables assigned to this class instance are managed by a single CachedEmbeddingBag.
    Those parameters in TablewiseEmbeddingBagConfig are ignored: cuda_row_num, buffer_size, initial_weight.
    """

    def __init__(self,
                 embedding_bag_config_list: List[TablewiseEmbeddingBagConfig],
                 embedding_dim: int,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.,
                 scale_grad_by_freq=False,
                 sparse=False,
                 _weight=None,
                 mode='mean',
                 include_last_offset=False,
                 dtype=None,
                 device=None,
                 cache_ratio=0.01,
                 warmup_ratio=0.7,
                 buffer_size=50_000,
                 pin_weight=False,
                 evict_strategy: EvictionStrategy = EvictionStrategy.LFU):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.rank_of_tables = [config.assigned_rank for config in embedding_bag_config_list]
        self.global_table_num_embeddings_list = [config.num_embeddings for config in embedding_bag_config_list]
        self.global_tables_num = len(embedding_bag_config_list)
        self.global_tables_offsets = torch.cumsum(torch.tensor([0] + self.global_table_num_embeddings_list), 0).cuda()
        self.assigned_table_list: List[int] = []
        self.pg = ProcessGroup(tp_degree=self.world_size)
        self.num_embeddings = 0
        for i, rank in enumerate(self.rank_of_tables):
            if rank == self.rank:
                self.assigned_table_list.append(i)
                self.num_embeddings += self.global_table_num_embeddings_list[i]
        self.include_last_offset = include_last_offset

        ids_freq_mapping = []
        for config in embedding_bag_config_list:
            if config.assigned_rank == self.rank:
                if config.ids_freq_mapping != None:
                    ids_freq_mapping.extend(config.ids_freq_mapping)
                else:
                    ids_freq_mapping = None
                    break
        self.cache_ratio = cache_ratio
        # table-associate cache
        cuda_row_num = int(cache_ratio * self.num_embeddings)
        super(ParallelCachedEmbeddingBagTablewise,
              self).__init__(self.num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq,
                             sparse, _weight, mode, include_last_offset, dtype, device, cache_ratio, ids_freq_mapping,
                             warmup_ratio, buffer_size, pin_weight, evict_strategy)

        # for assigned tables reconnection:
        self.idx_offset_list = []
        offset_cumsum = 0
        for table_i, table_num_embeddings in enumerate(self.global_table_num_embeddings_list):
            if self.rank_of_tables[table_i] == self.rank:
                self.idx_offset_list.append(offset_cumsum)
            else:
                offset_cumsum += table_num_embeddings

        # prepare list shape for all_to_all output
        self.embedding_dim_per_rank = [0 for i in range(self.world_size)]
        for rank in self.rank_of_tables:
            self.embedding_dim_per_rank[rank] += embedding_dim

        self.cache_op = True

    def forward(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor = None,
        per_sample_weights=None,
        shape_hook=None,
        already_split_along_rank=True,
    ):
        if not already_split_along_rank:
            # not recommanded. it takes time.
            batch_size = (offsets.shape[0]) // self.global_tables_num
            local_indices, local_offsets, local_per_sample_weights = self.split_along_rank(
                batch_size, indices, offsets, per_sample_weights)
        else:
            # recommanded.
            batch_size = (offsets.shape[0]) // len(self.assigned_table_list)
            local_indices, local_offsets, local_per_sample_weights = indices, offsets, per_sample_weights
        if self.cache_op:
            with torch.no_grad():
                indices = self.cache_weight_mgr.prepare_ids(local_indices)
        local_output = F.embedding_bag(indices.cuda(), self.cache_weight_mgr.cuda_cached_weight, local_offsets,
                                       self.max_norm, self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse,
                                       local_per_sample_weights, self.include_last_offset, self.padding_idx)
        local_output = torch.cat(local_output.split(batch_size), 1)
        remains = batch_size % self.world_size
        scatter_strides = [batch_size // self.world_size + int(i < remains) for i in range(self.world_size)]
        output_full = dual_all_to_all_tablewise(local_output, self.pg, scatter_strides, self.embedding_dim_per_rank)
        if shape_hook is not None:
            output_full = shape_hook(output_full)
        return output_full

    def split_along_rank(self,
                         batch_size,
                         indices: torch.Tensor,
                         offsets: torch.Tensor = None,
                         per_sample_weights=None):
        '''
        if input indices and offsets haven't been splitted along assigned rank, this function will do it.
        it takes time. please consider splitting data during batch loading.
        '''
        local_indices_list: List(torch.Tensor) = []
        local_offsets_list: List(torch.Tensor) = []
        if per_sample_weights != None:
            local_per_sample_weights_list: List(torch.Tensor) = []

        offset_pre_end = 0    # local_offsets trick
        for i, handle_table in enumerate(self.assigned_table_list):
            indices_start_position = offsets[batch_size * handle_table]
            if (not self.include_last_offset) and (batch_size * (handle_table + 1) >= indices.shape[0]):
                # till-the-end special case
                indices_end_position = indices.shape[0]
            else:
                indices_end_position = offsets[batch_size * (handle_table + 1)]
            # alternative approach: reduce malloc
            '''
            # 1. local_indices_list:
            local_indices = indices.narrow(0, indices_start_position, indices_end_position - indices_start_position)
            torch.sub(local_indices, self.idx_offset_list[i], out=local_indices)
            local_indices_list.append(local_indices)
            # 2. local_offsets_list:
            if i + 1 == len(self.assigned_table_list):
                # till-the-end special case
                if not self.include_last_offset:
                    local_offsets = offsets.narrow(0, batch_size * handle_table, batch_size)
                else:
                    local_offsets = offsets.narrow(0, batch_size * handle_table, batch_size + 1)
                torch.add(local_offsets, offset_pre_end - offsets[batch_size * handle_table], out=local_offsets)
                local_offsets_list.append(local_offsets)
            else:
                temp_holder = offsets[batch_size * handle_table].item()
                local_offsets = offsets.narrow(0, batch_size * handle_table, batch_size)
                torch.add(local_offsets, offset_pre_end - offsets[batch_size * handle_table], out=local_offsets)
                offset_pre_end = offsets[batch_size * (handle_table + 1)] + offset_pre_end - temp_holder
                local_offsets_list.append(local_offsets)
            '''
            # 1. local_indices_list:
            local_indices_list.append(
                indices.narrow(0, indices_start_position,
                               indices_end_position - indices_start_position).sub(self.idx_offset_list[i]))
            # 2. local_offsets_list:
            if i + 1 == len(self.assigned_table_list):
                # till-the-end special case
                if not self.include_last_offset:
                    local_offsets = offsets.narrow(0, batch_size * handle_table,
                                                   batch_size).add(offset_pre_end - offsets[batch_size *
                                                                                            (handle_table)])
                else:
                    local_offsets = offsets.narrow(0, batch_size * handle_table, batch_size +
                                                   1).add(offset_pre_end - offsets[batch_size * (handle_table)])
                local_offsets_list.append(local_offsets)
            else:
                local_offsets = offsets.narrow(0, batch_size * handle_table, batch_size +
                                               1).add(offset_pre_end - offsets[batch_size * (handle_table)])
                offset_pre_end = local_offsets[-1]
                local_offsets_list.append(local_offsets[:-1])
            # 3. local_per_sample_weights_list:
            if per_sample_weights != None:
                local_per_sample_weights_list.append(per_sample_weights[indices_start_position:indices_end_position])
        local_indices = torch.cat(local_indices_list, 0)
        local_offsets = torch.cat(local_offsets_list, 0)
        local_per_sample_weights = None
        if per_sample_weights != None:
            local_per_sample_weights = torch.cat(local_per_sample_weights_list, 0)
        return local_indices, local_offsets, local_per_sample_weights

    def set_cache_op(self, cache_op: bool = True):
        self.cache_op = cache_op

    def print_comm_stats_(self):
        self.cache_weight_mgr.print_comm_stats()

    def element_size(self):
        return self.weight.element_size()
