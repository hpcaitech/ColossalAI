import numpy as np
import torch
from torch.profiler import record_function
from typing import List, Optional
from contexttimer import Timer
from .copyer import LimitBuffIndexCopyer
from enum import Enum
import sys


class EvictionStrategy(Enum):
    LFU = 1
    # dataset aware eviction strategy
    DATASET = 2


class CachedParamMgr(torch.nn.Module):
    """
    Manage Embedding Weights on CPU and CUDA memory uses a software cache.
    CPU maintains the entire original weight. 
    CUDA maintains a fraction of the weights used in the upcoming computation. The row number in CUDA is controlled by `cuda_row_num`.
    During training, GPU needs to transmit embedding rows between CPU and GPU.

    Args:
        weight (torch.Tensor): the weight of the Embedding layer.
        cuda_row_num (int, optional): the number of rows cached in CUDA memory. Defaults to 0.
        buffer_size (int, optional): the number of rows in a data transmitter buffer. Defaults to 50_000.
        pin_weight (bool, optional): use pin memory to store the cpu weight. If set `True`, the cpu memory usage will increase largely. Defaults to False. 
        evict_strategy (EvictionStrategy, optional): the eviction strategy. There are two options. `EvictionStrategy.LFU` uses the least frequently used cache. `EvictionStrategy.DATASET`: use the stats collected from the target dataset. It usually leads to less cpu-gpu communication volume.
        Default as  EvictionStrategy.DATASET.
        use_cpu_caching (bool, optional): use cpu to execute cache indexing. It is slower than use gpu.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        cuda_row_num: int = 0,
        buffer_size: int = 50_000,
        pin_weight: bool = False,
        evict_strategy: EvictionStrategy = EvictionStrategy.DATASET,
        use_cpu_caching=False,
    ) -> None:
        super(CachedParamMgr, self).__init__()
        self.buffer_size = buffer_size
        self.num_embeddings, self.embedding_dim = weight.shape
        self.cuda_row_num = cuda_row_num
        self._cuda_available_row_num = self.cuda_row_num
        self.pin_weight = pin_weight
        self.elem_size_in_byte = weight.element_size()

        self._cpu_caching = use_cpu_caching

        if self._cpu_caching:
            self._cache_dev = torch.device('cpu')
        else:
            self._cache_dev = torch.cuda.current_device()

        # weight configure
        self._init_weight(weight)

        # Perf log
        self.num_hits_history = []
        self.num_miss_history = []
        self.num_write_back_history = []
        self._reset_comm_stats()

        self._evict_strategy = evict_strategy

        if self._evict_strategy == EvictionStrategy.LFU:
            # cache_row_idx -> frequency, freq of the cache rows.
            # classic lfu cache. evict the minimal freq value row in cuda cache.
            if self._cpu_caching:
                self.freq_cnter = torch.empty(self.cuda_row_num, device=self._cache_dev,
                                              dtype=torch.long).fill_(sys.maxsize)

            else:
                self.register_buffer("freq_cnter",
                                     torch.empty(self.cuda_row_num, device=self._cache_dev,
                                                 dtype=torch.long).fill_(sys.maxsize),
                                     persistent=False)

    def _find_evict_gpu_idxs(self, evict_num: int) -> torch.Tensor:
        """_find_evict_gpu_idxs 

        Find the gpu idxs to be evicted, according to their freq.

        Args:
            evict_num (int): how many rows has to be evicted

        Returns:
            torch.Tensor: a list tensor (1D), contains the gpu_row_idxs.
        """
        if self._evict_strategy == EvictionStrategy.LFU:
            # find the minimal evict_num freq entries in cached_idx_map
            _, evict_gpu_row_idxs = torch.topk(self.freq_cnter, evict_num, largest=False)
            return evict_gpu_row_idxs
        elif self._evict_strategy == EvictionStrategy.DATASET:
            # cached_idx_map itself implies the priority of eviction.
            # The value of self.cached_idx_map represents cpu_row_idx.
            # The larger it is, the less frequently it will appear in the dataset,
            # and the higher its eviction priority will be.
            _, evict_gpu_row_idxs = torch.topk(self.cached_idx_map, evict_num, largest=True)
            return evict_gpu_row_idxs
        else:
            raise TypeError

    def _init_weight(self, weight):
        if self.cuda_row_num > 0:
            # Enable cache with introducing auxiliary data structures
            self.cuda_cached_weight = torch.nn.Parameter(
                torch.zeros(self.cuda_row_num,
                            self.embedding_dim,
                            device=torch.cuda.current_device(),
                            dtype=weight.dtype))

            # pin memory cpu for higher CPU-GPU copy bandwidth
            self.weight = weight.pin_memory() if self.pin_weight else weight
            # map original id to new id with respect to frequency
            # id -> cpu_row_idx

            if self._cpu_caching:
                self.idx_map = torch.arange(self.num_embeddings, dtype=torch.long, device=self._cache_dev)
                self.cached_idx_map = torch.empty(self.cuda_row_num, device=self._cache_dev, dtype=torch.long).fill_(-1)
                self.inverted_cached_idx = torch.zeros(self.num_embeddings, device=self._cache_dev,
                                                       dtype=torch.long).fill_(-1)
            else:
                self.register_buffer(
                    "idx_map",
                    torch.arange(self.num_embeddings, dtype=torch.long, device=self._cache_dev),
                    persistent=False,
                )

                # cached_idx_map: gpu_row_idx -> cpu_row_idx
                self.register_buffer("cached_idx_map",
                                     torch.empty(self.cuda_row_num, device=self._cache_dev, dtype=torch.long).fill_(-1),
                                     persistent=False)

                # cpu_row_id -> gpu_row_idx.
                # gpu_row_idx as -1 means cpu_row_id not in CUDA.
                self.register_buffer("inverted_cached_idx",
                                     torch.zeros(self.num_embeddings, device=self._cache_dev,
                                                 dtype=torch.long).fill_(-1),
                                     persistent=False)

                self.evict_backlist = torch.tensor([], device=self._cache_dev)

            # index copy buffer size should less than 10% of cuda weight.
            if self.buffer_size > 0:
                self.limit_buff_index_copyer = LimitBuffIndexCopyer(self.buffer_size)

        else:
            # Disable cache so that FreqCacheEmbedding is compatible with vanilla EmbeddingBag
            # self.weight = torch.nn.Parameter(weight)
            # self.cuda_cached_weight = self.weight
            raise NotImplementedError()

    def cpu_weight_data(self, row_idx: int) -> torch.Tensor:
        """
        access a row of CPU weight.

        Args:
            row_idx (int): the idx of rows

        Returns:
            torch.Tensor: a piece of memory in CPU weight corresponding to row id's payload. The tensor is 1-D.
        """

        return self.weight.data.view(-1).narrow(0,
                                                int(row_idx) * self.embedding_dim,
                                                self.embedding_dim).view(1, self.embedding_dim)

    @property
    def cuda_available_row_num(self):
        return self._cuda_available_row_num

    @torch.no_grad()
    def reorder(self, ids_freq_mapping: Optional[List[int]] = None, warmup_ratio=0.7):
        """reorder
        reorder the weight according to ids' frequency in dataset before training.
        Execute only once before training, also known as warmup phase.
        
        Note:
            If you would like to use the DATASET as the eviction strategy, you must call this function.

        Note:
            If you are use the LFU as the eviction strategy, you can skip this function. If you still use this function. It will initialize
            The frequency in LFU cache using the dataset statistics.

        Args:
            ids_freq_mapping (List[int]): a list, whose offset is id number, value is freq. if None then not reorder the cpu weight.
            warmup_ratio (float): the amount of chunks preloaded in cuda cache
        """
        # reorder phase: reorder the cpu weight according to their freq stats in the target dataset.
        # reorder only works for DATASET eviction strategy.

        if ids_freq_mapping is not None and not isinstance(ids_freq_mapping, torch.Tensor):
            ids_freq_mapping = torch.tensor(ids_freq_mapping)

        if self._evict_strategy == EvictionStrategy.DATASET:
            if ids_freq_mapping is not None:
                tmp_idx = torch.argsort(ids_freq_mapping, descending=True)
                sorted_idx = torch.argsort(tmp_idx)
                self.idx_map.data.copy_(sorted_idx)

        # warmup phase: copy #preload_row_num rows from cpu to gpu.
        preload_row_num = min(int(np.ceil(self.cuda_row_num * warmup_ratio)), self.num_embeddings)
        if preload_row_num > 0:
            with Timer() as timer:
                # extract rows from cpu weight
                if self._evict_strategy == EvictionStrategy.LFU and ids_freq_mapping is not None:
                    freq_value, preload_cpu_ids = torch.topk(ids_freq_mapping, preload_row_num, dim=0, largest=True)
                    preload_cuda_row_idxs = torch.arange(preload_row_num).to(self._cache_dev)
                else:
                    preload_cpu_ids = torch.arange(preload_row_num)
                    preload_cuda_row_idxs = preload_cpu_ids.to(self._cache_dev)

                if self.buffer_size > 0:
                    self.limit_buff_index_copyer.index_copy(0,
                                                            src_index=preload_cpu_ids,
                                                            tgt_index=preload_cuda_row_idxs.cuda(),
                                                            src=self.weight.view(self.num_embeddings, -1),
                                                            tgt=self.cuda_cached_weight.view(self.cuda_row_num, -1))
                else:
                    preload_rows = self.weight.view(self.num_embeddings, -1).index_select(0, preload_cpu_ids).cuda()
                    self.cuda_cached_weight.view(self.cuda_row_num, -1).index_copy_(0, preload_cuda_row_idxs.cuda(),
                                                                                    preload_rows)

                # update auxiliary info
                self.cached_idx_map[preload_cuda_row_idxs] = preload_cpu_ids.to(self._cache_dev)
                self.inverted_cached_idx[preload_cpu_ids] = preload_cuda_row_idxs
                self._cuda_available_row_num -= preload_row_num

                if self._evict_strategy == EvictionStrategy.LFU:
                    # if the ids_freq_mapping is not None, we initialize the embedding row's freq value in LFU as its freq in dataset.
                    if ids_freq_mapping is None:
                        self.freq_cnter.index_fill_(0, preload_cuda_row_idxs, 0)
                    else:
                        self.freq_cnter[preload_cuda_row_idxs] = freq_value.to(self._cache_dev)

            print(f'Cache warmup finished cost {timer.elapsed} sec.')

    def flush(self):
        """flush all CUDA rows to CPU.
        The function is usually called after training finished.
        """
        slots = torch.nonzero(self.cached_idx_map > -1).squeeze(1)
        row_ids = self.cached_idx_map[slots]
        rows = self.cuda_cached_weight.view(self.cuda_row_num, -1).index_select(0, slots.cuda()).cpu()
        self.weight.view(self.num_embeddings, -1).index_copy_(0, row_ids.cpu(), rows)
        self.cached_idx_map.index_fill_(0, slots, -1)
        self.inverted_cached_idx.index_fill_(0, row_ids, -1)
        self._cuda_available_row_num += slots.numel()

        if self._evict_strategy == EvictionStrategy.LFU:
            self.freq_cnter.fill_(sys.maxsize)
        assert self._cuda_available_row_num == self.cuda_row_num
        assert torch.all(self.inverted_cached_idx == -1).item()
        assert torch.all(self.cached_idx_map == -1).item()

    def print_comm_stats(self):
        if self._cuda_to_cpu_numel > 0:
            print(
                f"CUDA->CPU BWD {self._cuda_to_cpu_numel * self.elem_size_in_byte / 1e6 / self._cuda_to_cpu_elapse} MB/s {self._cuda_to_cpu_numel / 1e6} M elem"
            )
        if self._cpu_to_cuda_numel > 0:
            print(
                f"CPU->CUDA BWD {self._cpu_to_cuda_numel * self.elem_size_in_byte / 1e6 / self._cpu_to_cuda_elpase} MB/s {self._cpu_to_cuda_numel / 1e6} M elem"
            )

    @torch.no_grad()
    def _id_to_cached_cuda_id(self, ids: torch.Tensor) -> torch.Tensor:
        """
        convert ids to indices in self.cuda_cached_weight.
        Implemented with parallel operations on GPU.

        Args:
            ids (torch.Tensor): ids from the dataset

        Returns:
            torch.Tensor: contains indices in self.cuda_cached_weight
        """
        ids = self.idx_map.index_select(0, ids.view(-1))
        ret = self.inverted_cached_idx.index_select(0, ids)
        return ret

    @torch.no_grad()
    def prepare_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """
        move the cpu embedding rows w.r.t. ids into CUDA memory

        Args:
            ids (torch.Tensor): the ids to be computed
        Returns:
            torch.Tensor: indices on the cuda_cached_weight.
        """
        with record_function("(zhg) get unique indices"):
            ids = ids.to(self._cache_dev)
            cpu_row_idxs, repeat_times = torch.unique(self.idx_map.index_select(0, ids), return_counts=True)

            assert len(cpu_row_idxs) <= self.cuda_row_num, \
                f"You move {len(cpu_row_idxs)} embedding rows from CPU to CUDA. " \
                f"It is larger than the capacity of the cache, which at most contains {self.cuda_row_num} rows, " \
                f"Please increase cuda_row_num or decrease the training batch size."
            self.evict_backlist = cpu_row_idxs

        with record_function("(zhg) get cpu row idxs"):
            comm_cpu_row_idxs = cpu_row_idxs[torch.isin(cpu_row_idxs, self.cached_idx_map, invert=True)]

        self.num_hits_history.append(len(cpu_row_idxs) - len(comm_cpu_row_idxs))
        self.num_miss_history.append(len(comm_cpu_row_idxs))
        self.num_write_back_history.append(0)

        # move sure the cuda rows will not be evicted!
        with record_function("(zhg) cache update"):
            self._prepare_rows_on_cuda(comm_cpu_row_idxs)

        self.evict_backlist = torch.tensor([], device=cpu_row_idxs.device, dtype=cpu_row_idxs.dtype)

        with record_function("(zhg) embed cpu rows idx -> cache gpu row idxs"):
            gpu_row_idxs = self._id_to_cached_cuda_id(ids)

        # update for LFU.
        if self._evict_strategy == EvictionStrategy.LFU:
            unique_gpu_row_idxs = self.inverted_cached_idx[cpu_row_idxs]
            self.freq_cnter.scatter_add_(0, unique_gpu_row_idxs, repeat_times)

        return gpu_row_idxs

    def _reset_comm_stats(self):
        self._cpu_to_cuda_numel = 0
        self._cpu_to_cuda_elpase = 0
        self._cuda_to_cpu_elapse = 0
        self._cuda_to_cpu_numel = 0

    def _row_in_cuda(self, row_id: int) -> bool:
        return self.inverted_cached_idx[row_id] != -1

    @torch.no_grad()
    def _prepare_rows_on_cuda(self, cpu_row_idxs: torch.Tensor) -> None:
        """prepare rows in cpu_row_idxs on CUDA memory

        Args:
            cpu_row_idxs (torch.Tensor): the rows to be placed on CUDA
        """
        evict_num = cpu_row_idxs.numel() - self.cuda_available_row_num
        if evict_num > 0:
            with Timer() as timer:
                mask_cpu_row_idx = torch.isin(self.cached_idx_map, self.evict_backlist)
                invalid_idxs = torch.nonzero(mask_cpu_row_idx).squeeze(1)
                if self._evict_strategy == EvictionStrategy.DATASET:
                    # mask method.
                    # set cached_idx_map[invalid_idxs] to -2.
                    # so those idxs will be sorted to end, therefore not being chosen as victim
                    backup_idxs = self.cached_idx_map[mask_cpu_row_idx].clone()
                    self.cached_idx_map.index_fill_(0, invalid_idxs, -2)
                    evict_gpu_row_idxs = self._find_evict_gpu_idxs(evict_num)
                    self.cached_idx_map.index_copy_(0, invalid_idxs, backup_idxs)

                elif self._evict_strategy == EvictionStrategy.LFU:
                    backup_freqs = self.freq_cnter[invalid_idxs].clone()
                    self.freq_cnter.index_fill_(0, invalid_idxs, sys.maxsize)
                    evict_gpu_row_idxs = self._find_evict_gpu_idxs(evict_num)
                    self.freq_cnter.index_copy_(0, invalid_idxs, backup_freqs)

                evict_info = self.cached_idx_map[evict_gpu_row_idxs]

                if self.buffer_size > 0:
                    self.limit_buff_index_copyer.index_copy(0,
                                                            src_index=evict_gpu_row_idxs,
                                                            tgt_index=evict_info.cpu(),
                                                            src=self.cuda_cached_weight.view(self.cuda_row_num, -1),
                                                            tgt=self.weight.view(self.num_embeddings, -1))
                else:
                    # allocate tmp memory on CPU and copy rows on CUDA to CPU.
                    rows = self.cuda_cached_weight.view(self.cuda_row_num,
                                                        -1).index_select(0, evict_gpu_row_idxs.cuda()).cpu()
                    self.weight.view(self.num_embeddings, -1).index_copy_(0, evict_info.cpu(), rows)

                self.cached_idx_map.index_fill_(0, evict_gpu_row_idxs, -1)
                self.inverted_cached_idx.index_fill_(0, evict_info, -1)
                # self.freq_cnter.index_fill(0, evict_gpu_row_idxs, sys.maxsize) # unnecessary
                self._cuda_available_row_num += evict_num

                weight_size = evict_gpu_row_idxs.numel() * self.embedding_dim
            self._cuda_to_cpu_elapse += timer.elapsed
            self._cuda_to_cpu_numel += weight_size
            # print(f"evict embedding weight: {weight_size*self.elem_size_in_byte/1e6:.2f} MB")

        with Timer() as timer:
            slots = torch.nonzero(self.cached_idx_map == -1).squeeze(1)[:cpu_row_idxs.numel()]
            # Here also allocate extra memory on CUDA. #cpu_row_idxs
            if self.buffer_size > 0:
                self.limit_buff_index_copyer.index_copy(0,
                                                        src_index=cpu_row_idxs.cpu(),
                                                        tgt_index=slots.cuda(),
                                                        src=self.weight.view(self.num_embeddings, -1),
                                                        tgt=self.cuda_cached_weight.view(self.cuda_row_num, -1))
            else:
                rows = self.weight.view(self.num_embeddings, -1).index_select(0, cpu_row_idxs.cpu()).cuda()
                self.cuda_cached_weight.view(self.cuda_row_num, -1).index_copy_(0, slots.cuda(), rows)
            slot_offsets = slots
            self.cached_idx_map[slots] = cpu_row_idxs
            self.inverted_cached_idx.index_copy_(0, cpu_row_idxs, slot_offsets)
            if self._evict_strategy == EvictionStrategy.LFU:
                self.freq_cnter.index_fill_(0, slots, 0)
            self._cuda_available_row_num -= cpu_row_idxs.numel()
        self._cpu_to_cuda_elpase += timer.elapsed
        weight_size = cpu_row_idxs.numel() * self.embedding_dim
        self._cpu_to_cuda_numel += weight_size
        # print(f"admit embedding weight: {weight_size*self.elem_size_in_byte/1e6:.2f} MB")

    def _find_free_cuda_row(self) -> int:
        if self._cuda_available_row_num == 0:
            return -1
        candidates = torch.nonzero(self.cached_idx_map == -1).squeeze(1)
        return candidates[0].item()

    def _evict(self) -> int:
        """
        deprecated

        evict one row from cuda to cpu.
        Returns: 
        (int) : the slot id be evicted.
        """
        mask = torch.logical_or(torch.isin(self.cached_idx_map, self.evict_backlist), self.cached_idx_map == -1)
        buf = self.cached_idx_map[mask].clone()
        idx = torch.nonzero(mask).squeeze(1)
        self.cached_idx_map.index_fill_(0, idx, -1)
        max_row, max_cpu_row_idx = torch.max(self.cached_idx_map, dim=0)
        max_gpu_row_idx = self.cached_idx_map[max_cpu_row_idx]

        if max_gpu_row_idx == -1:
            raise RuntimeError("Can not evict a row")

        max_gpu_row_idx = max_gpu_row_idx.item()
        max_offset = self.inverted_cached_idx[max_gpu_row_idx]
        # recover
        self.cached_idx_map.index_copy_(0, idx, buf)

        with Timer() as timer:
            cuda_tensor = torch.narrow(self.cuda_cached_weight.view(-1), 0, max_offset * self.embedding_dim,
                                       self.embedding_dim).view(1, self.embedding_dim)
            self.cpu_weight_data(max_gpu_row_idx).data.copy_(cuda_tensor)

        # update inverted_cached_idx, min_slot_id is evicted from cuda
        self.cached_idx_map[max_cpu_row_idx] = -1
        if self._evict_strategy == EvictionStrategy.LFU:
            self.freq_cnter[max_cpu_row_idx] = sys.maxsize
        self.inverted_cached_idx[max_gpu_row_idx] = -1

        self._cuda_available_row_num += 1

        self._cuda_to_cpu_numel += self.embedding_dim
        self._cuda_to_cpu_elapse += timer.elapsed
        # self.num_write_back_history[-1] += 1
        return max_cpu_row_idx

    @torch.no_grad()
    def _admit(self, row_id: int):
        """
        deprecated

        move in row_id to CUDA

        Args:
            row_id (int): the id of row to be moved in
        """
        # find a free slot in partial cuda weight
        slot_id = self._find_free_cuda_row()

        if slot_id == -1:
            # evict one row
            slot_id = self._evict()
        slot_offset = slot_id
        # copy payload from cpu to cuda
        with Timer() as timer:
            cuda_tensor = torch.narrow(self.cuda_cached_weight.view(-1), 0, slot_offset * self.embedding_dim,
                                       self.embedding_dim).view(1, self.embedding_dim)
            cuda_tensor.data.copy_(self.cpu_weight_data(row_id))

        # update the inverted_cached_idx
        self.cached_idx_map[slot_id] = row_id
        if self._evict_strategy == EvictionStrategy.LFU:
            self.freq_cnter[slot_id] = 0
        self.inverted_cached_idx[row_id] = slot_offset

        self._cuda_available_row_num -= 1

        self._cpu_to_cuda_numel += self.embedding_dim
        self._cpu_to_cuda_elpase += timer.elapsed
