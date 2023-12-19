from typing import List

import torch
from transformers.configuration_utils import PretrainedConfig

from colossalai.inference.config import InferenceConfig
from colossalai.inference.kv_cache import KVCacheManager
from colossalai.inference.logit_processors import logit_processor
from colossalai.inference.sampler import *
from colossalai.inference.struct import BatchInfo, Sequence


class RunningList:
    """
    RunningList is an structure for recording the running sequences, contains prefill and decoding list.
    Prefilling samples will be hold until the actual ratio of prefill samples versus decoding samples exceeds ratio.

    Args:
        prefill_ratio: (float) A ratio for determing whether to perform prefill or not.
        prefill: (List) List that contains default inputs, defaults to [].
    """

    def __init__(self, prefill_ratio: str, prefill: List[Sequence] = []):
        self.prefill_ratio = prefill_ratio
        self.decoding: List[Sequence] = []
        self.prefill: List[Sequence] = prefill

    def append(self, seq: Sequence):
        # add seq to prefilling list first.
        self.prefill.append(seq)

    def find_seq(self, request_id):
        for seq in self.decoding:
            if request_id == seq.request_id:
                return seq
        for seq in self.prefill:
            if request_id == seq.request_id:
                return seq
        return None

    def remove(self, seq):
        try:
            self.decoding.remove(seq)
        except:
            self.prefill.remove(seq)

    def ready_for_prefill(self):
        if not self.decoding:
            return len(self.prefill) > 0
        return len(self.prefill) / len(self.decoding) >= self.ratio

    def is_empty(self):
        return not self.decoding and not self.prefill


class RequestHandler:
    """
    RequestHandler is the core for handling existing requests and updating current batch.
    During generation process, we call schedule function each iteration to update current batch.

    Args:
       inference_config: Configuration for initialize and manage kv cache.
       model_config: Configuration for model
    """

    def __init__(self, inference_config: InferenceConfig, model_config: PretrainedConfig) -> None:
        self.inference_config = inference_config
        self._init_cache(model_config)

        self.running_list: RunningList = RunningList(inference_config.prefill_ratio)
        self.waiting_list: List[List] = [[], [], []]
        self.done_list: List[Sequence] = []
        self.running_batch = BatchInfo(is_prompts=False)
        self.prefill_batch = BatchInfo(is_prompts=True)

    def _init_cache(self, model_config):
        self.cache_manager = KVCacheManager(self.inference_config, model_config)

    def _has_waiting(self) -> bool:
        return any(lst for lst in self.waiting_list)

    def schedule(self):
        """
        The main logic of request handler.
        """
        if self._has_waiting():
            # Try to allocate cache blocks for the sequence using a priority of prompt length.
            for lst in reversed(self.waiting_list):
                if lst:
                    for seq in lst:
                        if seq.prompt_len > self.inference_config.max_input_len:
                            # If the prompt length is longer than max_input_len, abort the sequence.
                            self.abort_sequence(seq.request_id)
                            break
                        # Try to allocate cache blocks for the sequence.
                        if self.cache_manager.check_allocation(seq):
                            # If succeed, add the sequence to running list.
                            self.running_list.append(seq)
                            self.cache_manager.allocate_context_from_block_table(seq.block_table, seq.prompt_len)
                            lst.remove(seq)

        if self.running_list.ready_for_prefill():
            for seq in self.running_list.prefill:
                seq.mark_running()
            self.prefill_batch.init_batch(self.running_list.prefill)
            return self.prefill_batch

        return self.running_batch

    def add_sequence(self, req: Sequence):
        """
        Add the request to waiting list.
        """
        assert not self._find_sequence(req.request_id), f"Sequence {req.request_id} already exists."
        assert (
            req.prompt_len < self.inference_config.max_input_len
        ), f"Sequence {req.request_id} exceeds input length limit"

        self.waiting_list[req.prompt_len * 3 // self.inference_config.max_input_len].append(req)

    def abort_sequence(self, request_id: str):
        """
        Abort the request.
        """
        seq, priority = self._find_sequence(request_id)
        if seq.status.is_waiting:
            seq.mark_aborted()
            self.waiting_list[priority].remove(seq)
        elif seq.status.is_running():
            self.cache_manager.free_block_table(seq.block_table)
            self.running_list.remove(seq)
        else:
            try:
                self.done_list.remove(seq)
            except:
                return

    def _find_sequence(self, request_id: str) -> Sequence:
        """
        Find the request by request_id.
        """
        for priority, lst in enumerate(self.waiting_list):
            for seq in lst:
                if seq.request_id == request_id:
                    return seq, priority

        if self.running_list.find_seq(request_id):
            return seq, None

        return None

    def _sample(self, probs: torch.Tensor, logprobs: torch.Tensor, generation_config):
        if generation_config.num_beams == 1:
            if generation_config.do_sample:
                sample_tokens = greedy_sample(generation_config, logprobs)
            else:
                sample_tokens = multinomial_sample(generation_config, probs)
        else:
            sample_tokens = beam_search_sample(generation_config, logprobs, is_prompt=not self.prefill_batch.is_empty)

        return sample_tokens

    def mark_finished(self, sequence: Sequence, generation_config):
        if (
            sequence.output_token_id[-1] == generation_config.eos_id
            or sequence.output_len >= generation_config.max_output_len
        ):
            sequence.mark_finished()

    def check_unfinished_seqs(self) -> bool:
        return self._has_waiting() or not self.running_list.is_empty()

    def search_tokens(self, generation_config, logits):
        """
        Sample tokens for finished requests.
        """
        # do logit processor
        # NOTE: need to decide the granularity to process logits (sequence or batch)
        for type in ["top_p", "top_k", "min_p"]:
            if type in generation_config:
                logits = logit_processor(type, logits)

        # calculate probs
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # sample the next tokens
        sample_tokens = self._sample(probs, logprobs, generation_config)
        self.running_batch.update_batch_tokens(sample_tokens)

    def update(self):
        """
        Update current running list and done list
        """
        if not self.prefill_batch.is_empty:
            self.running_list.decoding.extend(self.running_list.prefill)
            self.running_batch.add_seqs(self.running_list.prefill)
            self.running_list.prefill.clear()
            self.prefill_batch.clear_batch()

        for seq in self.running_batch.sequences_set:
            if seq.check_finish():
                self.done_list.append(seq)
                self.running_list.remove(seq)
                self.running_batch.sequences_set.remove(seq)
                self.cache_manager.free_block_table(seq.block_table)

        return self.done_list
