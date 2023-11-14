# Adapted from https://github.com/ModelTC/lightllm

import collections
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from colossalai.inference.tensor_parallel import MemoryManager


# make batch infer state an attr of InferBatch
class InferSamplingParams:
    def __init__(
        self,
        do_sample: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        vocab_size: int = -1,
    ) -> None:
        self.do_sample = do_sample
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if self.top_k == -1:
            self.top_k = vocab_size
        return


@dataclass
class InferBatch:
    batch_id: int
    requests: List
    requests_idx_mapping: Dict[int, int]

    input_ids: torch.Tensor

    all_input_ids: List[List[int]]
    input_lengths: List[int]

    out_token_id_counts: List
    sampling_param_list: List[InferSamplingParams]

    nopad_total_token_num: int
    nopad_max_len_in_batch: int
    nopad_b_loc: torch.Tensor
    nopad_b_start_loc: torch.Tensor
    nopad_b_seq_len: torch.Tensor
    cache_manager: MemoryManager
    max_total_len: int

    @classmethod
    @torch.no_grad()
    def init_batch(
        cls,
        batch_id,
        requests,
        dtype: torch.dtype,
        device: torch.device,
        cache_manager: MemoryManager,
        vocab_size: int,
        max_total_len: int,
    ) -> "InferBatch":
        input_lengths = []
        all_input_ids = []
        requests_idx_mapping = {}

        out_token_id_counts = []
        sampling_param_list = []

        nopad_total_token_num = 0
        nopad_max_len_in_batch = 0
        nopad_b_loc = torch.empty((len(requests), max_total_len + 12), dtype=torch.long, device="cuda")
        # to avoid memory leak , we pre-allocate 12 more space for each batch.
        nopad_b_start_loc = torch.zeros(len(requests), dtype=torch.int32, device="cuda")
        for i, r in enumerate(requests):
            # request id -> idx in list mapping
            requests_idx_mapping[r["request_id"]] = i

            tokenized_input = r["input_id"]

            input_length = len(tokenized_input)
            input_lengths.append(input_length)
            all_input_ids.append(tokenized_input)
            out_token_id_counts.append(collections.defaultdict(int))

            # postprocessor
            sampling_param = r["sampling_param"]
            sampling_param["vocab_size"] = vocab_size
            sampling_param_list.append(InferSamplingParams(**sampling_param))

            nopad_total_token_num += input_length
            nopad_max_len_in_batch = max(nopad_max_len_in_batch, input_length)

        nopad_b_seq_len = torch.tensor(input_lengths, dtype=torch.int32, device="cuda")
        nopad_b_start_loc[1:] = torch.cumsum(nopad_b_seq_len, dim=0, dtype=torch.int32)[0:-1]

        if len(requests) > 1:
            input_ids = np.concatenate(all_input_ids, dtype=np.int64)
        else:
            input_ids = all_input_ids[0]

        # Create tensors on device
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)

        return cls(
            batch_id=batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            input_lengths=input_lengths,
            all_input_ids=all_input_ids,
            nopad_total_token_num=nopad_total_token_num,
            nopad_max_len_in_batch=nopad_max_len_in_batch,
            nopad_b_loc=nopad_b_loc,
            nopad_b_start_loc=nopad_b_start_loc,
            nopad_b_seq_len=nopad_b_seq_len,
            out_token_id_counts=out_token_id_counts,
            sampling_param_list=sampling_param_list,
            cache_manager=cache_manager,
            max_total_len=max_total_len,
        )

    @torch.no_grad()
    def free_self(self) -> None:
        """
        Free the memory of the InferBatch itself
        """
        remove_index = []
        for idx in range(len(self)):
            remove_index.append(
                self.nopad_b_loc[
                    idx,
                    (self.nopad_max_len_in_batch - 1)
                    - (self.nopad_b_seq_len[idx] - 1) : (self.nopad_max_len_in_batch - 1),
                ]
            )
        remove_index = torch.cat(remove_index, dim=-1)
        self.cache_manager.free(remove_index)

    @torch.no_grad()
    def filter(self, request_ids: List[int]) -> "InferBatch":
        """
        Filter finished batch and return a new InferBatch with left ones.
        """
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            return self
        requests_idx_mapping = {}
        indices = []
        requests = []
        all_input_ids = []
        input_lengths = []
        nopad_total_token_num = 0
        nopad_max_len_in_batch = 0
        nopad_b_loc = torch.empty((len(request_ids), self.max_total_len + 12), dtype=torch.long, device="cuda")
        nopad_b_start_loc = torch.zeros(len(request_ids), dtype=torch.int32, device="cuda")
        nopad_b_seq_len = torch.zeros(len(request_ids), dtype=torch.int32, device="cuda")

        left_idx = []
        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            left_idx.append(idx)

        left_idx_set = set(left_idx)
        remove_index = []
        for idx in range(len(self)):
            if idx not in left_idx_set:
                remove_index.append(
                    self.nopad_b_loc[
                        idx,
                        (self.nopad_max_len_in_batch - 1)
                        - (self.nopad_b_seq_len[idx] - 1) : (self.nopad_max_len_in_batch - 1),
                    ]
                )
        remove_index = torch.cat(remove_index, dim=-1)
        self.cache_manager.free(remove_index)

        nopad_max_len_in_batch = 0
        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            indices.append(idx)

        nopad_b_seq_len[:] = self.nopad_b_seq_len[indices]
        nopad_max_len_in_batch = torch.max(nopad_b_seq_len).item()
        nopad_b_start_loc[1:] = torch.cumsum(nopad_b_seq_len, dim=0, dtype=torch.int32)[0:-1]
        nopad_total_token_num = torch.sum(nopad_b_seq_len).item()

        nopad_b_loc[:, 0 : (nopad_max_len_in_batch - 1)] = self.nopad_b_loc[
            indices,
            (self.nopad_max_len_in_batch - 1) - (nopad_max_len_in_batch - 1) : (self.nopad_max_len_in_batch - 1),
        ]
        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            requests_idx_mapping[request_id] = i
            requests.append(self.requests[idx])
            all_input_ids.append(self.all_input_ids[idx])
            input_lengths.append(self.input_lengths[idx])

        input_ids = self.input_ids[indices]

        return InferBatch(
            batch_id=self.batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            input_lengths=input_lengths,
            all_input_ids=all_input_ids,
            nopad_total_token_num=nopad_total_token_num,
            nopad_max_len_in_batch=nopad_max_len_in_batch,
            nopad_b_loc=nopad_b_loc,
            nopad_b_start_loc=nopad_b_start_loc,
            nopad_b_seq_len=nopad_b_seq_len,
            out_token_id_counts=[self.out_token_id_counts[_i] for _i in indices],
            sampling_param_list=[self.sampling_param_list[_i] for _i in indices],
            cache_manager=self.cache_manager,
            max_total_len=self.max_total_len,
        )

    @classmethod
    @torch.no_grad()
    def merge(cls, batch1, batch2) -> "InferBatch":
        """
        Return megerd new InferBatch
        """
        requests = batch1.requests + batch2.requests
        requests_idx_mapping = {}
        new_batch_size = len(batch1) + len(batch2)

        input_ids = batch1.input_ids.new_empty(new_batch_size)
        all_input_ids = []
        input_lengths = []
        out_token_id_counts = []
        sampling_param_list = []

        cumulative_batch_size = 0
        nopad_total_token_num = batch1.nopad_total_token_num + batch2.nopad_total_token_num
        nopad_max_len_in_batch = max(batch1.nopad_max_len_in_batch, batch2.nopad_max_len_in_batch)
        max_total_len = max(batch1.max_total_len, batch2.max_total_len)
        nopad_b_loc = torch.empty((new_batch_size, batch1.max_total_len + 12), dtype=torch.long, device="cuda")
        nopad_b_start_loc = torch.zeros(new_batch_size, dtype=torch.int32, device="cuda")
        nopad_b_seq_len = torch.zeros(new_batch_size, dtype=torch.int32, device="cuda")
        nopad_start_loc_len_temp = 0
        batches = [batch1, batch2]
        for i, batch in enumerate(batches):
            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + cumulative_batch_size
            start_index = cumulative_batch_size
            end_index = cumulative_batch_size + len(batch)
            input_ids[start_index:end_index] = batch.input_ids
            nopad_b_seq_len[start_index:end_index] = batch.nopad_b_seq_len
            nopad_b_start_loc[start_index:end_index] = batch.nopad_b_start_loc + nopad_start_loc_len_temp
            nopad_start_loc_len_temp = nopad_b_start_loc[end_index - 1] + nopad_b_seq_len[end_index - 1]
            nopad_b_loc[
                start_index:end_index,
                nopad_max_len_in_batch - batch.nopad_max_len_in_batch : nopad_max_len_in_batch - 1,
            ] = batch.nopad_b_loc[:, : batch.nopad_max_len_in_batch - 1]

            all_input_ids.extend(batch.all_input_ids)

            input_lengths.extend(batch.input_lengths)
            out_token_id_counts.extend(batch.out_token_id_counts)
            sampling_param_list.extend(batch.sampling_param_list)
            # Update
            cumulative_batch_size += len(batch)

        nopad_b_loc[:, nopad_max_len_in_batch - 1] = (
            nopad_total_token_num - new_batch_size + torch.arange(0, new_batch_size, dtype=torch.int32, device="cuda")
        )
        return InferBatch(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            input_lengths=input_lengths,
            all_input_ids=all_input_ids,
            nopad_total_token_num=nopad_total_token_num,
            nopad_max_len_in_batch=nopad_max_len_in_batch,
            nopad_b_loc=nopad_b_loc,
            nopad_b_start_loc=nopad_b_start_loc,
            nopad_b_seq_len=nopad_b_seq_len,
            out_token_id_counts=out_token_id_counts,
            sampling_param_list=sampling_param_list,
            cache_manager=batches[0].cache_manager,
            max_total_len=max_total_len,
        )

    def __len__(self):
        return len(self.requests)

    def get_post_sample_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        presence_penalties: List[float] = []
        frequency_penalties: List[float] = []
        temperatures: List[float] = []
        top_ps: List[float] = []
        top_ks: List[int] = []
        p_token_ids: List[int] = []
        p_token_counts: List[int] = []
        p_seq_len: List[int] = [
            0,
        ]
        p_max_len_in_batch: int = 0
        for i, id_to_count in enumerate(self.out_token_id_counts):
            sample_param = self.sampling_param_list[i]
            presence_penalties.append(sample_param.presence_penalty)
            frequency_penalties.append(sample_param.frequency_penalty)
            temperatures.append(sample_param.temperature)
            top_ps.append(sample_param.top_p)
            top_ks.append(sample_param.top_k)

            for token_id, count in id_to_count.items():
                p_token_ids.append(token_id)
                p_token_counts.append(count)
            p_seq_len.append(len(id_to_count))
            p_max_len_in_batch = max(p_max_len_in_batch, len(id_to_count))

        presence_penalties = torch.tensor(presence_penalties, dtype=torch.float, device="cuda")
        frequency_penalties = torch.tensor(frequency_penalties, dtype=torch.float, device="cuda")
        temperatures = torch.tensor(temperatures, dtype=torch.float, device="cuda")
        top_ps = torch.tensor(top_ps, dtype=torch.float, device="cuda")
        top_ks = torch.tensor(top_ks, dtype=torch.int32, device="cuda")
        p_token_ids = torch.tensor(p_token_ids, dtype=torch.int32, device="cuda")
        p_token_counts = torch.tensor(p_token_counts, dtype=torch.int32, device="cuda")
        p_seq_len = torch.tensor(p_seq_len, dtype=torch.int32, device="cuda")
        p_cumsum_seq_len = torch.cumsum(p_seq_len, dim=0, dtype=torch.int32)
        return (
            presence_penalties,
            frequency_penalties,
            temperatures,
            top_ps,
            top_ks,
            p_token_ids,
            p_token_counts,
            p_cumsum_seq_len,
            p_max_len_in_batch,
        )
