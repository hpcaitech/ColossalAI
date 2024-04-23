# Adapted from https://github.com/ModelTC/lightllm

import uuid
from typing import List

import numpy as np

from .io_struct import Batch, Req


class ReqQueue:
    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size, waiting_req_list=[]) -> None:
        self.max_total_tokens = max_total_tokens
        assert batch_max_tokens is not None
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        self.waiting_req_list: List[Req] = waiting_req_list

    def append(self, req):
        self.waiting_req_list.append(req)
        return

    def _init_cache_list(self, current_batch: Batch):
        if current_batch is not None:
            self.cache_len_list = [
                (req.input_len + len(req.output_ids), req.max_output_len - len(req.output_ids) - 1)
                for req in current_batch.reqs
            ]
        else:
            self.cache_len_list = []

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req):
        self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1))  # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        # NOTE: change here < to <=
        return need_max_token_num <= self.max_total_tokens and len(self.cache_len_list) <= self.running_max_req_size

    def generate_new_batch(self, current_batch: Batch = None):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        self._init_cache_list(current_batch)
        can_run_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        for req in self.waiting_req_list:
            flag = self._can_add_new_req(req)
            if req.aborted:
                aborted_count += 1
                continue
            if flag and new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count :]
            return new_batch
        else:
            return None

    def __len__(self):
        return self.waiting_req_list.__len__()
