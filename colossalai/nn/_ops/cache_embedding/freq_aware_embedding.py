import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Iterator, Tuple

from .base_embeddings import BaseEmbeddingBag
from .chunk_param_mgr import CachedParamMgr
from torch.nn.parameter import Parameter
from recsys.utils import get_partition
from ..functional import dual_all_to_all


class FreqAwareEmbeddingBag(BaseEmbeddingBag):

    def __init__(self, num_embeddings, embedding_dim, dtype=None, *args, **kwargs):
        super(FreqAwareEmbeddingBag, self).__init__(num_embeddings, embedding_dim, *args, **kwargs)
        self._weight = torch.randn(self.num_embeddings, self.embedding_dim, device='cpu', dtype=dtype)

    def preprocess(self,
                   chunk_size: int,
                   cuda_row_num: int,
                   ids_freq_mapping: Optional[List[int]] = None,
                   warmup_ratio=0.7,
                   buffer_size=50_000):
        """
        Called after initialized. 
        Reorder the weight rows according to the ids_freq_mapping.
        Then, let the weights of the Module be managed by a CachedParamMgr.
        Args:
            chunk_size (int): chunk size
            cuda_row_num (int): number of rows can be hosted in CUDA memory
            ids_freq_mapping (List[int]): a list, idx is id number, value is freq
            warmup_ratio (float): the amount of rows preloaded in cuda cache
        """
        self.chunk_weight_mgr = CachedParamMgr(self._weight, cuda_row_num, buffer_size)
        self.chunk_weight_mgr.reorder(ids_freq_mapping, warmup_ratio)

    def forward(self, indices, offsets=None, per_sample_weights=None):
        with torch.no_grad():
            reorder_ids = self.chunk_weight_mgr.prepare_ids(indices)

        embeddings = F.embedding_bag(reorder_ids, self.chunk_weight_mgr.cuda_cached_weight, offsets, self.max_norm,
                                     self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse,
                                     per_sample_weights, self.include_last_offset, self.padding_idx)

        return embeddings

    @property
    def weight(self):
        assert self.chunk_weight_mgr is not None
        return self.chunk_weight_mgr.cpu_weight.narrow(0, 0, self.num_embeddings)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        yield 'weight', self.chunk_weight_mgr.cuda_cached_weight

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield self.chunk_weight_mgr.cuda_cached_weight

    @property
    def num_hits_history(self):
        return self.chunk_weight_mgr.num_hits_history

    @property
    def num_miss_history(self):
        return self.chunk_weight_mgr.num_miss_history

    @property
    def num_write_back_history(self):
        return self.chunk_weight_mgr.num_write_back_history

    @property
    def swap_in_bandwidth(self):
        if self.chunk_weight_mgr._cpu_to_cuda_numel > 0:
            return self.chunk_weight_mgr._cpu_to_cuda_numel * self.chunk_weight_mgr.elem_size_in_byte / 1e6 / \
                   self.chunk_weight_mgr._cpu_to_cuda_elpase
        else:
            return 0

    @property
    def swap_out_bandwidth(self):
        if self.chunk_weight_mgr._cuda_to_cpu_numel > 0:
            return self.chunk_weight_mgr._cuda_to_cpu_numel * self.chunk_weight_mgr.elem_size_in_byte / 1e6 / \
                   self.chunk_weight_mgr._cuda_to_cpu_elapse
        return 0

    @property
    def input_id_percent_in_load_chunk(self):
        return 0    # np.mean(self.chunk_weight_mgr.input_id_percent_in_load_chunk) * 100


class ParallelFreqAwareEmbeddingBag(BaseEmbeddingBag):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.,
                 scale_grad_by_freq=False,
                 sparse=False,
                 _weight=None,
                 mode='mean',
                 include_last_offset=False,
                 parallel_mode=ParallelMode.DEFAULT,
                 dtype=None,
                 debug=True):
        super(ParallelFreqAwareEmbeddingBag,
              self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq,
                             sparse, mode, include_last_offset)

        self.parallel_mode = parallel_mode
        self.debug = debug
        self.rank = DISTMGR.get_rank(self.parallel_mode)
        self.world_size = DISTMGR.get_world_size(self.parallel_mode)

        self.partition_start_index, self.partition_end_index, divisible = get_partition(
            embedding_dim, self.rank, self.world_size)
        self.embedding_dim_per_partition = self.partition_end_index - self.partition_start_index

        if _weight is None:
            self._weight = torch.empty(self.num_embeddings, self.embedding_dim_per_partition, device='cpu', dtype=dtype)
            self.init_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim]
            partition = torch.tensor_split(_weight, self.world_size, 1)[self.rank]
            assert list(partition.shape) == [num_embeddings, self.embedding_dim_per_partition]
            self._weight = partition

    @property
    def weight(self):
        return self.chunk_weight_mgr.cpu_weight

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        yield 'weight', self.chunk_weight_mgr.cuda_cached_weight

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield self.chunk_weight_mgr.cuda_cached_weight

    @torch.no_grad()
    def init_parameters(self):
        self._weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        if self.padding_idx is not None:
            self._weight[self.padding_idx].fill_(0)

    def preprocess(self,
                   chunk_size: int,
                   cuda_row_num: int,
                   ids_freq_mapping: Optional[List[int]] = None,
                   warmup_ratio: float = 0.7,
                   buffer_size: int = 50_000):
        self.chunk_weight_mgr = CachedParamMgr(self._weight, cuda_row_num, buffer_size=buffer_size)
        self.chunk_weight_mgr.reorder(ids_freq_mapping, warmup_ratio)

    def forward(self, indices, offsets=None, per_sample_weights=None, shape_hook=None, scatter_dim=0, gather_dim=-1):
        with torch.no_grad():
            reorder_ids = self.chunk_weight_mgr.prepare_ids(indices)

        output_shard = F.embedding_bag(reorder_ids, self.chunk_weight_mgr.cuda_cached_weight, offsets, self.max_norm,
                                       self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse,
                                       per_sample_weights, self.include_last_offset, self.padding_idx)

        if shape_hook is not None:
            output_shard = shape_hook(output_shard)

        output_full = dual_all_to_all(output_shard, self.parallel_mode, scatter_dim=scatter_dim, gather_dim=gather_dim)
        return output_full

    @classmethod
    def from_pretrained(cls,
                        embedding: torch.Tensor,
                        freeze: bool = True,
                        padding_idx: Optional[int] = None,
                        max_norm: Optional[float] = None,
                        norm_type: float = 2.,
                        scale_grad_by_freq: bool = False,
                        sparse: bool = False,
                        mode: str = 'mean',
                        include_last_offset: bool = False,
                        parallel_mode: ParallelMode = ParallelMode.DEFAULT,
                        debug: bool = True,
                        chunk_size: int = 16,
                        cuda_row_num: int = 100_000,
                        ids_freq_mapping: Optional[List[int]] = None,
                        warmup_ratio: float = 0.7) -> 'ParallelFreqAwareEmbeddingBag':
        rows, cols = embedding.shape
        embedding_bag = cls(rows, cols, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, embedding.cpu(),
                            mode, include_last_offset, parallel_mode, debug)
        embedding_bag.preprocess(chunk_size, cuda_row_num, ids_freq_mapping, warmup_ratio)
        embedding_bag.chunk_weight_mgr.cuda_cached_weight.requires_grad_ = not freeze
        return embedding_bag
