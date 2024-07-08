# might want to consider combine with InferenceConfig in colossalai/ppinference/inference_config.py later
from dataclasses import dataclass

import torch
from transformers.tokenization_utils_base import BatchEncoding

from .kvcache_manager import MemoryManager


# adapted from: lightllm/server/router/model_infer/infer_batch.py
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
    past_key_values_len: int = None

    is_context_stage: bool = False
    context_mem_index: torch.Tensor = None
    decode_is_contiguous: bool = None
    decode_mem_start: int = None
    decode_mem_end: int = None
    decode_mem_index: torch.Tensor = None
    decode_layer_id: int = None

    device: torch.device = torch.device("cuda")

    @property
    def total_token_num(self):
        # return self.batch_size * self.max_len_in_batch
        assert self.seq_len is not None and self.seq_len.size(0) > 0
        return int(torch.sum(self.seq_len))

    def set_cache_manager(self, manager: MemoryManager):
        self.cache_manager = manager

    # adapted from: https://github.com/ModelTC/lightllm/blob/28c1267cfca536b7b4f28e921e03de735b003039/lightllm/common/infer_utils.py#L1
    @staticmethod
    def init_block_loc(
        b_loc: torch.Tensor, seq_len: torch.Tensor, max_len_in_batch: int, alloc_mem_index: torch.Tensor
    ):
        """in-place update block loc mapping based on the sequence length of the inputs in current bath"""
        start_index = 0
        seq_len_numpy = seq_len.cpu().numpy()
        for i, cur_seq_len in enumerate(seq_len_numpy):
            b_loc[i, max_len_in_batch - cur_seq_len : max_len_in_batch] = alloc_mem_index[
                start_index : start_index + cur_seq_len
            ]
            start_index += cur_seq_len
        return

    @classmethod
    def init_from_batch(
        cls,
        batch: torch.Tensor,
        max_input_len: int,
        max_output_len: int,
        cache_manager: MemoryManager,
    ):
        if not isinstance(batch, (BatchEncoding, dict, list, torch.Tensor)):
            raise TypeError(f"batch type {type(batch)} is not supported in prepare_batch_state")

        input_ids_list = None
        attention_mask = None

        if isinstance(batch, (BatchEncoding, dict)):
            input_ids_list = batch["input_ids"]
            attention_mask = batch["attention_mask"]
        else:
            input_ids_list = batch
        if isinstance(input_ids_list[0], int):  # for a single input
            input_ids_list = [input_ids_list]
            attention_mask = [attention_mask] if attention_mask is not None else attention_mask

        batch_size = len(input_ids_list)

        seq_start_indexes = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        seq_lengths = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        start_index = 0

        max_len_in_batch = -1
        if isinstance(batch, (BatchEncoding, dict)):
            for i, attn_mask in enumerate(attention_mask):
                curr_seq_len = len(attn_mask)
                seq_lengths[i] = curr_seq_len
                seq_start_indexes[i] = start_index
                start_index += curr_seq_len
                max_len_in_batch = curr_seq_len if curr_seq_len > max_len_in_batch else max_len_in_batch
        else:
            length = max(len(input_id) for input_id in input_ids_list)
            for i, input_ids in enumerate(input_ids_list):
                curr_seq_len = length
                seq_lengths[i] = curr_seq_len
                seq_start_indexes[i] = start_index
                start_index += curr_seq_len
                max_len_in_batch = curr_seq_len if curr_seq_len > max_len_in_batch else max_len_in_batch
        block_loc = torch.zeros((batch_size, max_input_len + max_output_len), dtype=torch.long, device="cuda")

        return cls(
            batch_size=batch_size,
            max_len_in_batch=max_len_in_batch,
            seq_len=seq_lengths.to("cuda"),
            start_loc=seq_start_indexes.to("cuda"),
            block_loc=block_loc,
            decode_layer_id=0,
            past_key_values_len=0,
            is_context_stage=True,
            cache_manager=cache_manager,
        )
