"""
Refered/Modified from lightllm/common/mem_manager.py
of the ModelTC/lightllm GitHub repository
https://github.com/ModelTC/lightllm/blob/050af3ce65edca617e2f30ec2479397d5bb248c9/lightllm/common/mem_manager.py
we slightly changed it to make it suitable for our colossal-ai shardformer TP-engine design.
"""

import torch
from transformers.utils import logging


class MemoryManager:
    r"""
    Manage token block indexes and allocate physical memory for key and value cache

    Args:
        size: maximum token number used as the size of key and value buffer
        dtype: data type of cached key and value
        head_num: number of heads the memory manager is responsible for
        head_dim: embedded size per head
        layer_num: the number of layers in the model
        device: device used to store the key and value cache
    """

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: torch.device = torch.device("cuda"),
    ):
        self.logger = logging.get_logger(__name__)
        self.available_size = size
        self.max_len_in_batch = 0
        self._init_mem_states(size, device)
        self._init_kv_buffers(size, device, dtype, head_num, head_dim, layer_num)

    def _init_mem_states(self, size, device):
        """Initialize tensors used to manage memory states"""
        self.mem_state = torch.ones((size,), dtype=torch.bool, device=device)
        self.mem_cum_sum = torch.empty((size,), dtype=torch.int32, device=device)
        self.indexes = torch.arange(0, size, dtype=torch.long, device=device)

    def _init_kv_buffers(self, size, device, dtype, head_num, head_dim, layer_num):
        """Initialize key buffer and value buffer on specified device"""
        self.key_buffer = [
            torch.empty((size, head_num, head_dim), dtype=dtype, device=device) for _ in range(layer_num)
        ]
        self.value_buffer = [
            torch.empty((size, head_num, head_dim), dtype=dtype, device=device) for _ in range(layer_num)
        ]

    @torch.no_grad()
    def alloc(self, required_size):
        """allocate space of required_size by providing indexes representing available physical spaces"""
        if required_size > self.available_size:
            self.logger.warning(f"No enough cache: required_size {required_size} " f"left_size {self.available_size}")
            return None
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self.mem_cum_sum)
        select_index = torch.logical_and(self.mem_cum_sum <= required_size, self.mem_state == 1)
        select_index = self.indexes[select_index]
        self.mem_state[select_index] = 0
        self.available_size -= len(select_index)
        return select_index

    @torch.no_grad()
    def alloc_contiguous(self, required_size):
        """allocate contiguous space of required_size"""
        if required_size > self.available_size:
            self.logger.warning(f"No enough cache: required_size {required_size} " f"left_size {self.available_size}")
            return None
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self.mem_cum_sum)
        sum_size = len(self.mem_cum_sum)
        loc_sums = (
            self.mem_cum_sum[required_size - 1 :]
            - self.mem_cum_sum[0 : sum_size - required_size + 1]
            + self.mem_state[0 : sum_size - required_size + 1]
        )
        can_used_loc = self.indexes[0 : sum_size - required_size + 1][loc_sums == required_size]
        if can_used_loc.shape[0] == 0:
            self.logger.info(
                f"No enough contiguous cache: required_size {required_size} " f"left_size {self.available_size}"
            )
            return None
        start_loc = can_used_loc[0]
        select_index = self.indexes[start_loc : start_loc + required_size]
        self.mem_state[select_index] = 0
        self.available_size -= len(select_index)
        start = start_loc.item()
        end = start + required_size
        return select_index, start, end

    @torch.no_grad()
    def free(self, free_index):
        """free memory by updating memory states based on given indexes"""
        self.available_size += free_index.shape[0]
        self.mem_state[free_index] = 1

    @torch.no_grad()
    def free_all(self):
        """free all memory by updating memory states"""
        self.available_size = len(self.mem_state)
        self.mem_state[:] = 1
        self.max_len_in_batch = 0
        self.logger.info("freed all space of memory manager")
