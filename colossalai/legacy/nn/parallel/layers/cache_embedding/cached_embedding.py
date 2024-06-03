from typing import Iterator, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .base_embedding import BaseEmbeddingBag
from .cache_mgr import CachedParamMgr, EvictionStrategy


class CachedEmbeddingBag(BaseEmbeddingBag):
    """CachedEmbeddingBag

    Cached Embedding. Apply a GPU-based software cache approaches to dynamically manage the embedding table in the CPU and GPU memory space.
    It can leverage the id's frequency statistics of the target dataset, by passing a frequency list to param `ids_freq_mapping`.
    You can also apply a naive LFU cache eviction strategy by setting `evict_strategy` as EvictionStrategy.LFU.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int):  the size of each embedding vector
        padding_idx (int, optional): If specified, the entries at padding_idx do not contribute to the gradient; therefore, the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”. For a newly constructed EmbeddingBag, the embedding vector at padding_idx will default to all zeros, but can be updated to another value to be used as the padding vector. Note that the embedding vector at padding_idx is excluded from the reduction.
        max_norm (float, optional): If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm
        norm_type (str, optional): The p of the p-norm to compute for the max_norm option. Defaults to 2.
        scale_grad_by_freq (bool, optional): if given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default False. Note: this option is not supported when mode="max". Defaults to False.
        sparse (bool, optional): if True, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for more details regarding sparse gradients. Note: this option is not supported when mode="max".. Defaults to False.
        _weight (torch.Tensor, optional): an embedding weight tensor. Concatenate multiple tables in a embedding bag as a single one. Defaults to None.
        mode (str, optional): "sum", "mean" or "max". Specifies the way to reduce the bag. "sum" computes the weighted sum, taking per_sample_weights into consideration. "mean" computes the average of the values in the bag, "max" computes the max value over each bag. Default: "mean". Defaults to 'mean'.
        include_last_offset (bool, optional): if True, offsets has one additional element, where the last element is equivalent to the size of indices. This matches the CSR format.. Defaults to False.
        dtype (torch.dtype, optional): data type of the cpu weight initialization. Defaults to None meaning float32.
        device (torch.device, optional): device type to the cpu weight. Defaults to None meaning cpu.
        cache_ratio (float, float): cache ratio of the #cuda_weight_row / #cpu_weight_row
        ids_freq_mapping (Union[List, torch.Tensor], optional): the frequency of each embedding vector occurs in dataset. Defaults to None.
        warmup_ratio (float, optional): the ratio of cuda cache is warmuped with. Defaults to 0.7.
        buffer_size (int, optional): the max number of vectors in transmitter buffer. If set to 0, the buffer is not used. Defaults to 0.
        pin_weight (bool, optional): pin the cpu weight. Defaults to False.
        evict_strategy (EvictionStrategy, optional): evict strategy of the software cache. Defaults to EvictionStrategy.DATASET.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[torch.Tensor] = None,
        mode: str = "mean",
        include_last_offset: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        cache_ratio: float = 0.01,
        ids_freq_mapping: Optional[Union[List, torch.Tensor]] = None,
        warmup_ratio: float = 0.7,
        buffer_size: int = 0,
        pin_weight: bool = False,
        evict_strategy: EvictionStrategy = EvictionStrategy.LFU,
    ):
        super(CachedEmbeddingBag, self).__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            mode,
            include_last_offset,
        )

        assert cache_ratio <= 1.0, f"cache ratio {cache_ratio} must less than 1.0"
        self.evict_strategy = evict_strategy
        if _weight is None:
            _weight = self._weight_alloc(dtype, device)
        cuda_row_num = int(num_embeddings * cache_ratio)
        # configure weight & cache
        self._preprocess(_weight, cuda_row_num, ids_freq_mapping, warmup_ratio, buffer_size, pin_weight)
        self.cache_op = True

    def set_cache_mgr_async_copy(self, flag):
        self.cache_weight_mgr._async_copy = flag

    def _weight_alloc(self, dtype, device):
        weight = torch.empty(self.num_embeddings, self.embedding_dim, dtype=dtype, device=device)
        with torch.no_grad():
            weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
            if self.padding_idx is not None:
                weight[self.padding_idx].fill_(0)
        return weight

    def _preprocess(
        self,
        weight,
        cuda_row_num: int,
        ids_freq_mapping: Optional[List[int]] = None,
        warmup_ratio=0.7,
        buffer_size=50_000,
        pin_weight=False,
    ):
        """
        Called after initialized.
        Reorder the weight rows according to the ids_freq_mapping.
        Then, let the weights of the Module be managed by a CachedParamMgr.

        Args:
            cuda_row_num (int): number of rows can be hosted in CUDA memory
            ids_freq_mapping (List[int]): a list, idx is id number, value is freq
            warmup_ratio (float): the amount of rows preloaded in cuda cache
        """
        self.cache_weight_mgr = CachedParamMgr(
            weight, cuda_row_num, buffer_size, pin_weight, evict_strategy=self.evict_strategy
        )
        self.cache_weight_mgr.reorder(ids_freq_mapping, warmup_ratio)

    def forward(self, input, offsets=None, per_sample_weights=None, shape_hook=None):
        if self.cache_op:
            with torch.no_grad():
                input = self.cache_weight_mgr.prepare_ids(input)

        embeddings = F.embedding_bag(
            input.cuda(),
            self.cache_weight_mgr.cuda_cached_weight,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights,
            self.include_last_offset,
            self.padding_idx,
        )
        if shape_hook is not None:
            embeddings = shape_hook(embeddings)
        return embeddings

    @property
    def weight(self):
        return self.cache_weight_mgr.weight

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        yield "weight", self.cache_weight_mgr.cuda_cached_weight

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield self.cache_weight_mgr.cuda_cached_weight

    def set_cache_op(self, cache_op: bool = True):
        self.cache_op = cache_op

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
            return (
                self.cache_weight_mgr._cpu_to_cuda_numel
                * self.cache_weight_mgr.elem_size_in_byte
                / 1e6
                / self.cache_weight_mgr._cpu_to_cuda_elapse
            )
        else:
            return 0

    @property
    def swap_out_bandwidth(self):
        if self.cache_weight_mgr._cuda_to_cpu_numel > 0:
            return (
                self.cache_weight_mgr._cuda_to_cpu_numel
                * self.cache_weight_mgr.elem_size_in_byte
                / 1e6
                / self.cache_weight_mgr._cuda_to_cpu_elapse
            )
        return 0
