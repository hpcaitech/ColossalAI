import torch
import torch.nn.functional as F
from typing import List, Optional, Iterator, Tuple

from .base_embedding import BaseEmbeddingBag
from .cache_mgr import CachedParamMgr, EvictionStrategy
from torch.nn.parameter import Parameter


class FreqAwareEmbeddingBag(BaseEmbeddingBag):

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
                 dtype=None,
                 device=None,
                 cuda_row_num=0,
                 ids_freq_mapping=None,
                 warmup_ratio=0.7,
                 buffer_size=50_000,
                 pin_weight=False,
                 evict_strategy: EvictionStrategy = EvictionStrategy.DATASET):
        super(FreqAwareEmbeddingBag, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                                                    scale_grad_by_freq, sparse, mode, include_last_offset)

        self.evict_strategy = evict_strategy
        if _weight is None:
            _weight = self._weight_alloc(dtype, device)

        # configure weight & cache
        self._preprocess(_weight, cuda_row_num, ids_freq_mapping, warmup_ratio, buffer_size, pin_weight)

    def _weight_alloc(self, dtype, device):
        weight = torch.empty(self.num_embeddings, self.embedding_dim, dtype=dtype, device=device)
        with torch.no_grad():
            weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
            if self.padding_idx is not None:
                weight[self.padding_idx].fill_(0)
        return weight

    def _preprocess(self,
                    weight,
                    cuda_row_num: int,
                    ids_freq_mapping: Optional[List[int]] = None,
                    warmup_ratio=0.7,
                    buffer_size=50_000,
                    pin_weight=False):
        """
        Called after initialized. 
        Reorder the weight rows according to the ids_freq_mapping.
        Then, let the weights of the Module be managed by a CachedParamMgr.
        
        Args:
            cuda_row_num (int): number of rows can be hosted in CUDA memory
            ids_freq_mapping (List[int]): a list, idx is id number, value is freq
            warmup_ratio (float): the amount of rows preloaded in cuda cache
        """
        self.cache_weight_mgr = CachedParamMgr(weight,
                                               cuda_row_num,
                                               buffer_size,
                                               pin_weight,
                                               evict_strategy=self.evict_strategy)
        self.cache_weight_mgr.reorder(ids_freq_mapping, warmup_ratio)

    def forward(self, indices, offsets=None, per_sample_weights=None, shape_hook=None):
        with torch.no_grad():
            reorder_ids = self.cache_weight_mgr.prepare_ids(indices)

        embeddings = F.embedding_bag(reorder_ids, self.cache_weight_mgr.cuda_cached_weight, offsets, self.max_norm,
                                     self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse,
                                     per_sample_weights, self.include_last_offset, self.padding_idx)
        if shape_hook is not None:
            embeddings = shape_hook(embeddings)
        return embeddings

    @property
    def weight(self):
        return self.cache_weight_mgr.weight

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        yield 'weight', self.cache_weight_mgr.cuda_cached_weight

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield self.cache_weight_mgr.cuda_cached_weight


############################# Perf Log ###################################

    @property
    def num_hits_history(self):
        return self.cache_weight_mgr.num_hits_history

    @property
    def num_miss_history(self):
        return self.cache_weight_mgr.num_miss_history

    @property
    def num_write_back_history(self):
        return self.cache_weight_mgr.num_write_back_history

    @property
    def swap_in_bandwidth(self):
        if self.cache_weight_mgr._cpu_to_cuda_numel > 0:
            return self.cache_weight_mgr._cpu_to_cuda_numel * self.cache_weight_mgr.elem_size_in_byte / 1e6 / \
                   self.cache_weight_mgr._cpu_to_cuda_elpase
        else:
            return 0

    @property
    def swap_out_bandwidth(self):
        if self.cache_weight_mgr._cuda_to_cpu_numel > 0:
            return self.cache_weight_mgr._cuda_to_cpu_numel * self.cache_weight_mgr.elem_size_in_byte / 1e6 / \
                   self.cache_weight_mgr._cuda_to_cpu_elapse
        return 0