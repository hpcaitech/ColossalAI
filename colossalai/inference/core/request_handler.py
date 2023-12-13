from typing import List

import torch
from colossalai.inference.inference_struct import BatchHandler, RequsetStatus, Sequence
from colossalai.inference.kv_cache import KVCacheManager
from colossalai.inference.logit_processors import logit_processor
from colossalai.inference.sampler import *

class RunningList:
    def __init__(self, ratio):
        self.ratio = ratio
        self.decoding = []
        self.prefill = []

    def append(self, seq):
        # add seq to prefilling list first.
        self.prefill.append(seq)

    def find_seq(self, seq_id):
        for seq in self.decoding:
            if seq_id == seq.seq_id:
                return seq
        for seq in self.prefill:
            if seq_id == seq.seq_id:
                return seq
        return None

    def remove(self, seq):
        self.decoding.remove(seq)
        self.prefill.remove(seq)

    def ready_for_prefill(self):
        return len(self.prefill) / len(self.decoding) >= self.ratio

    def is_empty(self):
        return not self.decoding and not self.prefill


class RequestHandler:
    """
    RequestHandler is the core for handling existing requests and updating current batch.
    During generation process, we call schedule function each iteration to update current batch.

    Args:
       inference_config: Configuration for initialize and manage kv cache.
    """

    def __init__(self, inference_config) -> None:
        self.inference_config = inference_config
        self._init_cache()

        self.running_list: RunningList = RunningList(inference_config.ratio)
        self.waiting_list: List[List] = [[], [], []]
        self.done_list: List[Sequence] = []
        self.batch_info = BatchHandler()

    def _init_cache(self, inference_config):
        self.cache_manager = KVCacheManager(inference_config)

    def _has_waiting(self) -> bool:
        return all(not lst for lst in self.waiting_list)

    def schedule(self):
        """
        The main logic of request handler.
        """

        if self._has_waiting():
            # Try to allocate cache blocks for the sequence using a priority of prompt length.
            for lst in reversed(self.waiting_list):
                if lst:
                    seq = lst[0]
                    if seq.prompt_len > self.inference_config.max_input_len:
                        # If the prompt length is longer than max_input_len, abort the sequence.
                        self.abort_sequence(seq.seq_id)
                        break
                    # Try to allocate cache blocks for the sequence.
                    if self.cache_manager.num_available_blocks > self.cache_manager.max_blocks_per_sequence:
                        # If succeed, add the sequence to running list.
                        self.running_list.append(seq)
                        self.cache_manager.allocate_context_from_block_table(seq.block_table_index)
                        lst.pop(0)

        self.batch_info.update_batch(self.running_list)

    def add_sequence(self, req: Sequence):
        """
        Add the request to waiting list.
        """
        assert not self._find_sequence(req.seq_id), f"Sequence {req.seq_id} already exists."
        self.waiting_list[req.prompt_len * 3 / self.inference_config.max_input_len].append(req)

    def abort_sequence(self, seq_id: str):
        """
        Abort the request.
        """
        seq = self._find_sequence(seq_id)
        self.cache_manager.free_block_table(seq.block_table)
        if seq.status == RequsetStatus.WAITING:
            seq.status = RequsetStatus.ABORTED
            self.waiting_list.remove(seq_id)  # maybe wrong
        else:
            self.running_list.remove(seq_id)

    def _find_sequence(self, seq_id: str) -> Sequence:
        """
        Find the request by seq_id.
        """
        for priority, lst in enumerate(self.waiting_list):
            for seq in lst:
                if seq.seq_id == seq_id:
                    return seq, priority

        if self.running_list.find_seq(seq_id):
            return seq

        return None

    def _sample(self,probs: torch.Tensor, logprobs: torch.Tensor,generation_config):
        if generation_config.num_beams == 1:
            if generation_config.do_sample:
                sample_tokens = greedy_sample()
            else:
                sample_tokens = multinomial_sample()
        else:
            sample_tokens = beam_search_sample()

        return sample_tokens

    def mark_finished(self,sequence:Sequence,generation_config):
        if sequence.output_token_id[-1] == generation_config.eos_id or  sequence.output_len >= generation_config.max_output_len: 
            sequence.mark_finished()
    
    def check_unfinished_seqs(self) -> bool:
        return self._has_waiting() or not self.running_list.is_empty()

    def search_tokens(self,generation_config,logits):
        """
	    Sample tokens for finished requests.
	    """ 
        # do logit processor 
        # NOTE: need to decide the granularity to process logits (sequence or batch)
        for type in ['top_p','top_k','min_p']:
            if type in generation_config:
                logits = logit_processor(type,logits)

        # calculate probs
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # sample the next tokens
        sample_tokens = self._sample(probs,logprobs,generation_config)

        for idx,sample in enumerate(sample_tokens):
            sequence =  self.batch_info.sequences_set[idx]
            sequence.append(sample)
            self.mark_finished(sequence,generation_config)

    def update(self):
        """
        Update current running list and done list
        """
        for seq in self.batch_info.sequences_set:
            if seq.check_finish():
                self.done_list.append(seq)
                self.running_list.remove(seq)
                self.cache_manager.free_block_table(seq.block_table_index)
        return self.done_list
