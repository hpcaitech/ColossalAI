import sys
from contextlib import contextmanager
from enum import Enum
from typing import List, Optional

import numpy as np
import torch
from contexttimer import Timer
from torch.profiler import record_function

from .copyer import LimitBuffIndexCopyer


class EvictionStrategy(Enum):
    LFU = 1
    # dataset aware eviction strategy
    DATASET = 2


def _wait_for_data(t, stream: Optional[torch.cuda.streams.Stream]) -> None:
    if stream is None:
        return
    torch.cuda.current_stream().wait_stream(stream)
    # As mentioned in https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html,
    # PyTorch uses the "caching allocator" for memory allocation for tensors. When a tensor is
    # freed, its memory is likely to be reused by newly constructed tensors.  By default,
    # this allocator traces whether a tensor is still in use by only the CUDA stream where it
    # was created.   When a tensor is used by additional CUDA streams, we need to call record_stream
    # to tell the allocator about all these streams.  Otherwise, the allocator might free the
    # underlying memory of the tensor once it is no longer used by the creator stream.  This is
    # a notable programming trick when we write programs using multi CUDA streams.
    cur_stream = torch.cuda.current_stream()
    assert isinstance(t, torch.Tensor)
    t.record_stream(cur_stream)


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
        evict_strategy (EvictionStrategy, optional): the eviction strategy. There are two options.
        `EvictionStrategy.LFU`: use the least frequently used cache.
        `EvictionStrategy.DATASET`: use the stats collected from the target dataset. It usually leads to less cpu-gpu communication volume.
        Defaults to EvictionStrategy.DATASET.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        cuda_row_num: int = 0,
        buffer_size: int = 0,
        pin_weight: bool = True,
        evict_strategy: EvictionStrategy = EvictionStrategy.DATASET,
        async_copy: bool = False,
    ) -> None:
        super(CachedParamMgr, self).__init__()
        self.buffer_size = buffer_size
        self.num_embeddings, self.embedding_dim = weight.shape
        self.cuda_row_num = cuda_row_num
        self._cuda_available_row_num = self.cuda_row_num
        self.pin_weight = pin_weight
        self.elem_size_in_byte = weight.element_size()

        # weight configure
        self._init_weight(weight)

        # Perf log
        self.num_hits_history = []
        self.num_miss_history = []
        self.num_write_back_history = []

        self._evict_strategy = evict_strategy

        self._async_copy = async_copy

        if self._async_copy:
            self._memcpy_stream = torch.cuda.Stream()

            print("use async copy")

        if self._evict_strategy == EvictionStrategy.LFU:
            # cache_row_idx -> frequency, freq of the cache rows.
            # classic lfu cache. evict the minimal freq value row in cuda cache.
            self.register_buffer(
                "freq_cnter",
                torch.empty(self.cuda_row_num, device=torch.cuda.current_device(), dtype=torch.long).fill_(sys.maxsize),
                persistent=False,
            )
        self._elapsed_dict = {}
        self._show_cache_miss = True
        self._reset_comm_stats()

    def _reset_comm_stats(self):
        for k in self._elapsed_dict.keys():
            self._elapsed_dict[k] = 0

        self._cpu_to_cuda_numel = 0
        self._cuda_to_cpu_numel = 0
        if self._show_cache_miss:
            self._cache_miss = 0
            self._total_cache = 0

    @contextmanager
    def timer(self, name):
        with Timer() as t:
            yield
            torch.cuda.synchronize()

        if name not in self._elapsed_dict.keys():
            self._elapsed_dict[name] = 0
        self._elapsed_dict[name] += t.elapsed

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
                torch.zeros(
                    self.cuda_row_num, self.embedding_dim, device=torch.cuda.current_device(), dtype=weight.dtype
                )
            )

            # pin memory cpu for higher CPU-GPU copy bandwidth
            self.weight = weight.pin_memory() if self.pin_weight else weight
            # map original id to new id with respect to frequency
            # id -> cpu_row_idx
            self.register_buffer(
                "idx_map",
                torch.arange(self.num_embeddings, dtype=torch.long, device=torch.cuda.current_device()),
                persistent=False,
            )

            # cached_idx_map: gpu_row_idx -> cpu_row_idx
            self.register_buffer(
                "cached_idx_map",
                torch.empty(self.cuda_row_num, device=torch.cuda.current_device(), dtype=torch.long).fill_(-1),
                persistent=False,
            )

            # cpu_row_id -> gpu_row_idx.
            # gpu_row_idx as -1 means cpu_row_id not in CUDA.
            self.register_buffer(
                "inverted_cached_idx",
                torch.zeros(self.num_embeddings, device=torch.cuda.current_device(), dtype=torch.long).fill_(-1),
                persistent=False,
            )

            self.evict_backlist = torch.tensor([], device=torch.cuda.current_device())

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

        return (
            self.weight.data.view(-1)
            .narrow(0, int(row_idx) * self.embedding_dim, self.embedding_dim)
            .view(1, self.embedding_dim)
        )

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
                    preload_cuda_row_idxs = torch.arange(preload_row_num).cuda()
                else:
                    preload_cpu_ids = torch.arange(preload_row_num)
                    preload_cuda_row_idxs = preload_cpu_ids.cuda()
                if self.buffer_size > 0:
                    self.limit_buff_index_copyer.index_copy(
                        0,
                        src_index=preload_cpu_ids,
                        tgt_index=preload_cuda_row_idxs,
                        src=self.weight.view(self.num_embeddings, -1),
                        tgt=self.cuda_cached_weight.view(self.cuda_row_num, -1),
                    )
                else:
                    preload_rows = self.weight.view(self.num_embeddings, -1).index_select(0, preload_cpu_ids).cuda()
                    self.cuda_cached_weight.view(self.cuda_row_num, -1).index_copy_(
                        0, preload_cuda_row_idxs, preload_rows
                    )

                # update auxiliary info
                self.cached_idx_map[preload_cuda_row_idxs] = preload_cpu_ids.cuda()
                self.inverted_cached_idx[preload_cpu_ids] = preload_cuda_row_idxs
                self._cuda_available_row_num -= preload_row_num

                if self._evict_strategy == EvictionStrategy.LFU:
                    # if the ids_freq_mapping is not None, we initialize the embedding row's freq value in LFU as its freq in dataset.
                    if ids_freq_mapping is None:
                        self.freq_cnter.index_fill_(0, preload_cuda_row_idxs, 0)
                    else:
                        self.freq_cnter[preload_cuda_row_idxs] = freq_value.cuda()

            print(f"Cache warmup finished cost {timer.elapsed} sec.")

    def flush(self):
        """flush all CUDA rows to CPU.
        The function is usually called after training finished.
        """
        slots = torch.nonzero(self.cached_idx_map > -1).squeeze(1)
        row_ids = self.cached_idx_map[slots]
        rows = self.cuda_cached_weight.view(self.cuda_row_num, -1).index_select(0, slots).cpu()
        self.weight.view(self.num_embeddings, -1).index_copy_(0, row_ids.cpu(), rows)
        self.cached_idx_map.index_fill_(0, slots, -1)
        self.inverted_cached_idx.index_fill_(0, row_ids, -1)
        self._cuda_available_row_num += slots.numel()

        if self._show_cache_miss:
            self._cache_miss = 0
            self._total_cache = 0

        if self._evict_strategy == EvictionStrategy.LFU:
            self.freq_cnter.fill_(sys.maxsize)
        assert self._cuda_available_row_num == self.cuda_row_num
        assert torch.all(self.inverted_cached_idx == -1).item()
        assert torch.all(self.cached_idx_map == -1).item()

    def print_comm_stats(self):
        if self._cuda_to_cpu_numel > 0 and "3_evict_out" in self._elapsed_dict:
            elapsed = self._elapsed_dict["3_evict_out"]
            print(
                f"CUDA->CPU BWD {self._cuda_to_cpu_numel * self.elem_size_in_byte / 1e6 / elapsed} MB/s {self._cuda_to_cpu_numel / 1e6} M elem"
            )
            print(f"cuda_to_cpu_elapse {elapsed} sec")
        if self._cpu_to_cuda_numel > 0 and "5_evict_in" in self._elapsed_dict:
            elapsed = self._elapsed_dict["5_evict_in"]
            print(
                f"CPU->CUDA BWD {self._cpu_to_cuda_numel * self.elem_size_in_byte / 1e6 / elapsed} MB/s {self._cpu_to_cuda_numel / 1e6} M elem"
            )
            print(f"cpu_to_cuda_elapse {elapsed} sec")

        for k, v in self._elapsed_dict.items():
            print(f"{k}: {v}")

        print(f"cache miss ratio {self._cache_miss / self._total_cache}")

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
        torch.cuda.synchronize()
        with self.timer("cache_op") as gtimer:
            # identify cpu rows to cache
            with self.timer("1_identify_cpu_row_idxs") as timer:
                with record_function("(cache) get unique indices"):
                    if self._evict_strategy == EvictionStrategy.LFU:
                        cpu_row_idxs, repeat_times = torch.unique(ids, return_counts=True)
                    else:
                        cpu_row_idxs, repeat_times = torch.unique(self.idx_map.index_select(0, ids), return_counts=True)

                    assert len(cpu_row_idxs) <= self.cuda_row_num, (
                        f"You move {len(cpu_row_idxs)} embedding rows from CPU to CUDA. "
                        f"It is larger than the capacity of the cache, which at most contains {self.cuda_row_num} rows, "
                        f"Please increase cuda_row_num or decrease the training batch size."
                    )
                    self.evict_backlist = cpu_row_idxs
                    tmp = torch.isin(cpu_row_idxs, self.cached_idx_map, invert=True)
                    comm_cpu_row_idxs = cpu_row_idxs[tmp]

                    if self._show_cache_miss:
                        self._cache_miss += torch.sum(repeat_times[tmp])
                        self._total_cache += ids.numel()

            self.num_hits_history.append(len(cpu_row_idxs) - len(comm_cpu_row_idxs))
            self.num_miss_history.append(len(comm_cpu_row_idxs))
            self.num_write_back_history.append(0)

            # move sure the cuda rows will not be evicted!
            with record_function("(cache) prepare_rows_on_cuda"):
                with self.timer("prepare_rows_on_cuda") as timer:
                    self._prepare_rows_on_cuda(comm_cpu_row_idxs)

            self.evict_backlist = torch.tensor([], device=cpu_row_idxs.device, dtype=cpu_row_idxs.dtype)

            with self.timer("6_update_cache") as timer:
                with record_function("6_update_cache"):
                    gpu_row_idxs = self._id_to_cached_cuda_id(ids)

                # update for LFU.
                if self._evict_strategy == EvictionStrategy.LFU:
                    unique_gpu_row_idxs = self.inverted_cached_idx[cpu_row_idxs]
                    self.freq_cnter.scatter_add_(0, unique_gpu_row_idxs, repeat_times)

        return gpu_row_idxs

    def _row_in_cuda(self, row_id: int) -> bool:
        return self.inverted_cached_idx[row_id] != -1

    @torch.no_grad()
    def _prepare_rows_on_cuda(self, cpu_row_idxs: torch.Tensor) -> None:
        """prepare rows in cpu_row_idxs on CUDA memory
        Args:
            cpu_row_idxs (torch.Tensor): the rows to be placed on CUDA
        """
        evict_num = cpu_row_idxs.numel() - self.cuda_available_row_num

        cpu_row_idxs_copy = cpu_row_idxs.cpu()

        # move evict in rows to gpu
        if self._async_copy:
            if self.buffer_size == 0:
                evict_in_rows_gpu = (
                    self.weight.view(self.num_embeddings, -1).index_select(0, cpu_row_idxs_copy).pin_memory()
                )
                with torch.cuda.stream(self._memcpy_stream):
                    evict_in_rows_gpu = evict_in_rows_gpu.to(torch.cuda.current_device(), non_blocking=True)
            else:
                raise NotImplemented

        if evict_num > 0:
            with self.timer("2_identify_cuda_row_idxs") as timer:
                mask_cpu_row_idx = torch.isin(self.cached_idx_map, self.evict_backlist)
                invalid_idxs = torch.nonzero(mask_cpu_row_idx).squeeze(1)
                if self._evict_strategy == EvictionStrategy.DATASET:
                    # mask method.
                    # set cached_idx_map[invalid_idxs] to -2.
                    # so those idxs will be sorted to end, therefore not being chosen as victim
                    backup_idxs = self.cached_idx_map[mask_cpu_row_idx].clone()
                    self.cached_idx_map.index_fill_(0, invalid_idxs, -2)

                    with self.timer("2_1_find_evict_gpu_idxs") as timer:
                        evict_gpu_row_idxs = self._find_evict_gpu_idxs(evict_num)

                    # move evict out rows to cpu
                    if self._async_copy:
                        evict_out_rows_gpu = self.cuda_cached_weight.view(self.cuda_row_num, -1).index_select(
                            0, evict_gpu_row_idxs
                        )
                        evict_out_rows_cpu = torch.empty_like(evict_out_rows_gpu, device="cpu", pin_memory=True)
                        with torch.cuda.stream(None):
                            evict_out_rows_cpu.copy_(evict_out_rows_gpu, non_blocking=True)
                    self.cached_idx_map.index_copy_(0, invalid_idxs, backup_idxs)

                elif self._evict_strategy == EvictionStrategy.LFU:
                    with self.timer("2_1_backup_freqs") as timer:
                        backup_freqs = self.freq_cnter[invalid_idxs].clone()
                        self.freq_cnter.index_fill_(0, invalid_idxs, sys.maxsize)

                    with self.timer("2_2_find_evict_gpu_idxs") as timer:
                        evict_gpu_row_idxs = self._find_evict_gpu_idxs(evict_num)

                    if self._async_copy:
                        evict_out_rows_gpu = self.cuda_cached_weight.view(self.cuda_row_num, -1).index_select(
                            0, evict_gpu_row_idxs
                        )
                        evict_out_rows_cpu = torch.empty_like(evict_out_rows_gpu, device="cpu", pin_memory=True)
                        with torch.cuda.stream(None):
                            evict_out_rows_cpu.copy_(evict_out_rows_gpu, non_blocking=True)

                    with self.timer("2_3_revert_freqs") as timer:
                        self.freq_cnter.index_copy_(0, invalid_idxs, backup_freqs)

                evict_info = self.cached_idx_map[evict_gpu_row_idxs]

            with self.timer("3_evict_out") as timer:
                if self.buffer_size > 0:
                    self.limit_buff_index_copyer.index_copy(
                        0,
                        src_index=evict_gpu_row_idxs,
                        tgt_index=evict_info.cpu(),
                        src=self.cuda_cached_weight.view(self.cuda_row_num, -1),
                        tgt=self.weight.view(self.num_embeddings, -1),
                    )
                else:
                    # allocate tmp memory on CPU and copy rows on CUDA to CPU.
                    # TODO async gpu -> cpu
                    if self._async_copy:
                        _wait_for_data(evict_out_rows_cpu, None)
                    else:
                        with self.timer("3_1_evict_out_index_select") as timer:
                            evict_out_rows_cpu = self.cuda_cached_weight.view(self.cuda_row_num, -1).index_select(
                                0, evict_gpu_row_idxs
                            )
                        with self.timer("3_2_evict_out_gpu_to_cpu_copy") as timer:
                            evict_out_rows_cpu = evict_out_rows_cpu.cpu()

                    with self.timer("3_2_evict_out_cpu_copy") as timer:
                        self.weight.view(self.num_embeddings, -1).index_copy_(0, evict_info.cpu(), evict_out_rows_cpu)

                self.cached_idx_map.index_fill_(0, evict_gpu_row_idxs, -1)
                self.inverted_cached_idx.index_fill_(0, evict_info, -1)
                # self.freq_cnter.index_fill(0, evict_gpu_row_idxs, sys.maxsize) # unnecessary
                self._cuda_available_row_num += evict_num

                weight_size = evict_gpu_row_idxs.numel() * self.embedding_dim
                self._cuda_to_cpu_numel += weight_size
            # print(f"evict embedding weight: {weight_size*self.elem_size_in_byte/1e6:.2f} MB")

        # slots of cuda weight to evict in
        with self.timer("4_identify_cuda_slot") as timer:
            slots = torch.nonzero(self.cached_idx_map == -1).squeeze(1)[: cpu_row_idxs.numel()]

        # TODO wait for optimize
        with self.timer("5_evict_in") as timer:
            # Here also allocate extra memory on CUDA. #cpu_row_idxs
            if self.buffer_size > 0:
                self.limit_buff_index_copyer.index_copy(
                    0,
                    src_index=cpu_row_idxs_copy,
                    tgt_index=slots,
                    src=self.weight.view(self.num_embeddings, -1),
                    tgt=self.cuda_cached_weight.view(self.cuda_row_num, -1),
                )
            else:
                if self._async_copy:
                    _wait_for_data(evict_in_rows_gpu, self._memcpy_stream)
                else:
                    with self.timer("5_1_evict_in_index_select") as timer:
                        # narrow index select to a subset of self.weight
                        # tmp = torch.narrow(self.weight.view(self.num_embeddings, -1), 0, min(cpu_row_idxs).cpu(), max(cpu_row_idxs) - min(cpu_row_idxs) + 1)
                        # evict_in_rows_gpu = tmp.index_select(0, cpu_row_idxs_copy - min(cpu_row_idxs).cpu())
                        evict_in_rows_gpu = (
                            self.weight.view(self.num_embeddings, -1).index_select(0, cpu_row_idxs_copy).pin_memory()
                        )

                    with self.timer("5_2_evict_in_gpu_to_cpu_copy") as timer:
                        evict_in_rows_gpu = evict_in_rows_gpu.cuda()

                    with self.timer("5_3_evict_in_index_copy") as timer:
                        self.cuda_cached_weight.view(self.cuda_row_num, -1).index_copy_(0, slots, evict_in_rows_gpu)

        with self.timer("6_update_cache") as timer:
            self.cached_idx_map[slots] = cpu_row_idxs
            self.inverted_cached_idx.index_copy_(0, cpu_row_idxs, slots)
            if self._evict_strategy == EvictionStrategy.LFU:
                self.freq_cnter.index_fill_(0, slots, 0)
            self._cuda_available_row_num -= cpu_row_idxs.numel()

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
            cuda_tensor = torch.narrow(
                self.cuda_cached_weight.view(-1), 0, max_offset * self.embedding_dim, self.embedding_dim
            ).view(1, self.embedding_dim)
            self.cpu_weight_data(max_gpu_row_idx).data.copy_(cuda_tensor)

        # update inverted_cached_idx, min_slot_id is evicted from cuda
        self.cached_idx_map[max_cpu_row_idx] = -1
        if self._evict_strategy == EvictionStrategy.LFU:
            self.freq_cnter[max_cpu_row_idx] = sys.maxsize
        self.inverted_cached_idx[max_gpu_row_idx] = -1

        self._cuda_available_row_num += 1

        self._cuda_to_cpu_numel += self.embedding_dim
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
            cuda_tensor = torch.narrow(
                self.cuda_cached_weight.view(-1), 0, slot_offset * self.embedding_dim, self.embedding_dim
            ).view(1, self.embedding_dim)
            cuda_tensor.data.copy_(self.cpu_weight_data(row_id))

        # update the inverted_cached_idx
        self.cached_idx_map[slot_id] = row_id
        if self._evict_strategy == EvictionStrategy.LFU:
            self.freq_cnter[slot_id] = 0
        self.inverted_cached_idx[row_id] = slot_offset

        self._cuda_available_row_num -= 1

        self._cpu_to_cuda_numel += self.embedding_dim
