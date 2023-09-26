import abc
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import record_function

from colossalai.legacy.nn._ops._utils import dual_all_to_all_tablewise
from colossalai.legacy.tensor import ProcessGroup

from .cache_mgr import EvictionStrategy
from .cached_embedding import CachedEmbeddingBag
from .embedding_config import TablewiseEmbeddingBagConfig


class ParallelCachedEmbeddingBagTablewiseSpiltCache(abc.ABC, nn.Module):
    """
    every table assigned to this class instance is managed by a CachedEmbeddingBag.
    """

    def __init__(
        self,
        embedding_bag_config_list: List[TablewiseEmbeddingBagConfig],
        embedding_dim: int,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        mode="mean",
        include_last_offset=False,
        dtype=None,
        device=None,
        warmup_ratio=0.7,
        pin_weight=False,
        evict_strategy: EvictionStrategy = EvictionStrategy.LFU,
    ):
        super(ParallelCachedEmbeddingBagTablewiseSpiltCache, self).__init__()
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

        # prepare CachedEmbeddingBag list

        self.cached_embedding_bag_list: nn.ModuleList = nn.ModuleList()
        for config in embedding_bag_config_list:
            if config.assigned_rank != self.rank:
                continue
            self.cached_embedding_bag_list.append(
                CachedEmbeddingBag(
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
                    cuda_row_num=config.cuda_row_num,
                    ids_freq_mapping=config.ids_freq_mapping,
                    warmup_ratio=warmup_ratio,
                    buffer_size=config.buffer_size,
                    pin_weight=pin_weight,
                    evict_strategy=evict_strategy,
                )
            )

        # prepare list shape for all_to_all output
        self.embedding_dim_per_rank = [0 for i in range(self.world_size)]
        for rank in self.rank_of_tables:
            self.embedding_dim_per_rank[rank] += embedding_dim

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor = None, per_sample_weights=None, shape_hook=None):
        # determine indices to handle
        batch_size = (offsets.shape[0]) // self.global_tables_num
        local_output_list = []
        for i, handle_table in enumerate(self.assigned_table_list):
            with record_function("(tablewise) prepare indices and offsets"):
                with record_function("part 1"):
                    indices_start_position = offsets[batch_size * handle_table]
                    if (not self.include_last_offset) and (batch_size * (handle_table + 1) >= indices.shape[0]):
                        # till the end special case
                        indices_end_position = indices.shape[0]
                    else:
                        indices_end_position = offsets[batch_size * (handle_table + 1)]
                with record_function("part 2"):
                    # local_indices = indices[indices_start_position:indices_end_position] - self.global_tables_offsets[handle_table]
                    local_indices = indices.narrow(
                        0, indices_start_position, indices_end_position - indices_start_position
                    ).sub(self.global_tables_offsets[handle_table])
                    if self.include_last_offset:
                        # local_offsets = offsets[batch_size * handle_table:batch_size * (handle_table + 1) + 1] - offsets[batch_size * (handle_table)]
                        local_offsets = offsets.narrow(0, batch_size * handle_table, batch_size + 1).sub(
                            offsets[batch_size * (handle_table)]
                        )
                    else:
                        # local_offsets = offsets[batch_size * handle_table:batch_size * (handle_table + 1)] - offsets[batch_size * (handle_table)]
                        local_offsets = offsets.narrow(0, batch_size * handle_table, batch_size).sub(
                            offsets[batch_size * (handle_table)]
                        )
                local_per_sample_weights = None
                if per_sample_weights != None:
                    local_per_sample_weights = per_sample_weights[indices_start_position:indices_end_position]
            with record_function("(tablewise) tablewise forward"):
                local_output_list.append(
                    self.cached_embedding_bag_list[i](local_indices, local_offsets, local_per_sample_weights)
                )

        # get result of shape = (batch_size, (len(assigned_table_list)*embedding_dim))
        local_output = torch.cat(local_output_list, 1)
        # then concatenate those local_output on the second dimension.
        # use all_to_all
        remains = batch_size % self.world_size
        scatter_strides = [batch_size // self.world_size + int(i < remains) for i in range(self.world_size)]
        output_full = dual_all_to_all_tablewise(local_output, self.pg, scatter_strides, self.embedding_dim_per_rank)
        if shape_hook is not None:
            output_full = shape_hook(output_full)
        return output_full

    def element_size(self):
        if len(self.assigned_table_list) == 0:
            return 0
        return self.cached_embedding_bag_list[0].cache_weight_mgr.weight.element_size()

    def print_comm_stats_(self):
        cuda_to_cpu_elem_num = 0
        cpu_to_cuda_elem_num = 0
        for cached_embedding_bag in self.cached_embedding_bag_list:
            cuda_to_cpu_elem_num += cached_embedding_bag.cache_weight_mgr._cuda_to_cpu_numel
            cpu_to_cuda_elem_num += cached_embedding_bag.cache_weight_mgr._cpu_to_cuda_numel
        print(f"CUDA->CPU num: {cuda_to_cpu_elem_num / 1e6} M elem")
        print(f"CPU->CUDA num: {cpu_to_cuda_elem_num / 1e6} M elem")
