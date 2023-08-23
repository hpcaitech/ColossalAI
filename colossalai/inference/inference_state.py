# might want to consider combine with InferenceConfig in colossalai/ppinference/inference_config.py later
from dataclasses import dataclass
from typing import Any

import torch

from colossalai.inference.kvcache_manager import MemoryManager


@dataclass
class BatchInferState:
    r"""
    Information to be passed and used for a batch of inputs during
    a single model forward
    """
    batch_size: int
    max_len_in_batch: int

    cache_manager: MemoryManager = None

    block_loc: torch.Tensor = None
    start_loc: torch.Tensor = None
    seq_len: torch.Tensor = None

    is_context_stage: bool = False
    context_mem_index: torch.Tensor = None
    decode_is_contiguous: bool = None
    decode_mem_start: int = None
    decode_mem_end: int = None
    decode_mem_index: torch.Tensor = None

    device: torch.device = torch.device('cuda')

    @property
    def total_token_num(self):
        return self.batch_size * self.max_len_in_batch

    def set_cache_manager(self, manager: MemoryManager):
        self.cache_manager = manager

    def step_inference_state(self):
        """ update indexes used for kv cache management at the end of model forward """
        self.start_loc = self.start_loc + torch.arange(0, self.batch_size, dtype=torch.int32, device=self.device)
        self.seq_len += 1

    @staticmethod
    def init_block_loc(b_loc: torch.Tensor, seq_len: torch.Tensor, max_len_in_batch: int,
                       alloc_mem_index: torch.Tensor):
        """ in-place update block loc mapping based on the sequence length of the inputs in current bath"""
        start_index = 0
        seq_len_numpy = seq_len.cpu().numpy()
        for i, cur_seq_len in enumerate(seq_len_numpy):
            b_loc[i, max_len_in_batch - cur_seq_len:max_len_in_batch] = alloc_mem_index[start_index:start_index +
                                                                                        cur_seq_len]
            start_index += cur_seq_len
        return
