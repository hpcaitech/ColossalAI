from typing import Dict, List, Union

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationConfig

from colossalai.inference.batch_bucket import BatchBucket
from colossalai.inference.config import InferenceConfig
from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.inference.kv_cache import KVCacheManager, RPCKVCacheManager
from colossalai.inference.struct import DiffusionSequence, RequestStatus, Sequence
from colossalai.logging import get_dist_logger

logger = get_dist_logger(__name__)

__all__ = ["RunningList", "RequestHandler"]


class RunningList:
    """
    RunningList is an structure for recording the running sequences, contains prefill and decoding list.
    Prefilling samples will be hold until the actual ratio of prefill samples versus decoding samples exceeds ratio.

    Args:
        prefill_ratio: (float) A ratio for determing whether to perform prefill or not.
        _prefill (OrderedDict[Sequence]): Mapping of sequence uid -> Sequence.
        _decoding (OrderedDict[Sequence]): Mapping of sequence uid -> Sequence.
    """

    def __init__(self, prefill_ratio: int, prefill: List[Sequence] = None) -> None:
        self.prefill_ratio = prefill_ratio
        self._decoding: Dict[int, Sequence] = dict()
        self._prefill: Dict[int, Sequence] = (
            dict({seq.request_id: seq for seq in self._prefill}) if prefill is not None else dict()
        )

    @property
    def decoding(self):
        return list(self._decoding.values())

    @property
    def prefill(self):
        return list(self._prefill.values())

    @property
    def prefill_seq_num(self):
        return len(self._prefill)

    @property
    def decoding_seq_num(self):
        return len(self._decoding)

    @property
    def total_seq_num(self):
        return self.prefill_seq_num + self.decoding_seq_num

    def append(self, seq: Sequence):
        assert (seq.request_id not in self._prefill) and (
            seq.request_id not in self._decoding
        ), f"Sequence uid {seq.request_id} already exists."
        self._prefill[seq.request_id] = seq

    def extend(self, seqs: List[Sequence]):
        for seq in seqs:
            self._prefill[seq.request_id] = seq

    def find_seq(self, request_id) -> Union[Sequence, None]:
        seq = None
        if request_id in self._decoding:
            seq = self._decoding[request_id]
        elif request_id in self._prefill:
            seq = self._prefill[request_id]
        return seq

    def remove(self, seq: Sequence) -> None:
        if seq.request_id in self._decoding:
            self._decoding.pop(seq.request_id)
        elif seq.request_id in self._prefill:
            self._prefill.pop(seq.request_id)
        else:
            raise ValueError(f"Sequence {seq.request_id} is not in running list")

    def ready_for_prefill(self):
        if not self._decoding:
            return len(self._prefill) > 0
        return len(self._prefill) / len(self._decoding) >= self.prefill_ratio

    def is_empty(self):
        return not self._decoding and not self._prefill

    def mark_prefill_running(self) -> None:
        for seq_id in self._prefill:
            self._prefill[seq_id].mark_running()

    def move_prefill_to_decoding(self, seq_ids: List[int]) -> None:
        for seq_id in seq_ids:
            assert seq_id in self._prefill, f"Sequence {seq_id} is not in prefill list"
            self._decoding[seq_id] = self._prefill.pop(seq_id)


class NaiveRequestHandler:
    def __init__(self) -> None:
        self.running_list: List[DiffusionSequence] = []
        self.waiting_list: List[str] = []

    def _has_waiting(self) -> bool:
        return any(lst for lst in self.waiting_list)

    def _has_running(self) -> bool:
        return any(lst for lst in self.running_list)

    def check_unfinished_reqs(self):
        return self._has_waiting() or self._has_running()

    def add_sequence(self, seq: DiffusionSequence):
        """
        Add the request to waiting list.
        """
        assert not self._find_sequence(seq.request_id), f"Sequence {seq.request_id} already exists."
        self.waiting_list.append(seq)

    def _find_sequence(self, request_id: int) -> DiffusionSequence:
        """
        Find the request by request_id.
        """
        for lst in enumerate(self.waiting_list + self.running_list):
            for seq in lst:
                if seq.request_id == request_id:
                    return seq
        return None

    def schedule(self):
        ret = None
        if self._has_waiting:
            ret = self.waiting_list[0]
            self.waiting_list = self.waiting_list[1:]
        return ret


class RequestHandler(NaiveRequestHandler):
    """
    RequestHandler is the core for handling existing requests and updating current batch.
    During generation process, we call schedule function each iteration to update current batch.

    Args:
       inference_config: Configuration for initialize and manage kv cache.
       model_config: Configuration for model
       dtype (torch.dtype): The data type for weights and activations.
    """

    def __init__(self, inference_config: InferenceConfig, model_config: PretrainedConfig) -> None:
        self.inference_config = inference_config
        self.running_list: RunningList = RunningList(inference_config.prefill_ratio)
        self.waiting_list: List[List] = [[], [], []]
        self.done_list: List[Sequence] = []
        self.dtype = inference_config.dtype
        self.max_batch_size = inference_config.max_batch_size

        # initialize cache
        self._init_cache(model_config)

        # initialize batch
        device = torch.cuda.current_device()
        kv_max_split_num = (
            inference_config.max_input_len + inference_config.max_output_len + inference_config.block_size - 1
        ) // inference_config.block_size
        head_dim = model_config.hidden_size // model_config.num_attention_heads

        fd_inter_tensor = FDIntermTensors()

        if fd_inter_tensor._tensors_initialized:
            fd_inter_tensor._reset()

        # For Spec-Dec, process the speculated tokens plus the token in the last step for each seq
        max_n_tokens = self.max_batch_size
        max_n_tokens *= self.inference_config.max_n_spec_tokens + 1

        fd_inter_tensor.initialize(
            max_batch_size=max_n_tokens,
            num_attn_heads=model_config.num_attention_heads // inference_config.tp_size,
            kv_max_split_num=kv_max_split_num,
            head_dim=head_dim,
            dtype=self.dtype,
            device=device,
        )

        # TODO In the continuous batching scenario, the batch size may be greater than max_batch_size,
        # which may cause bugs and this issue should be fixed later.
        self.running_bb = BatchBucket(
            num_heads=model_config.num_attention_heads // inference_config.tp_size,
            head_dim=head_dim,
            max_batch_size=self.max_batch_size,
            max_length=inference_config.max_input_len + inference_config.max_output_len,
            block_size=inference_config.block_size,
            kv_max_split_num=kv_max_split_num,
            fd_interm_tensor=fd_inter_tensor,
            dtype=self.dtype,
            device=device,
            enable_streamingllm=inference_config.enable_streamingllm,
            start_token_size=inference_config.start_token_size,
            generated_token_size=inference_config.generated_token_size,
        )
        self.prefill_bb = BatchBucket(
            num_heads=model_config.num_attention_heads // inference_config.tp_size,
            head_dim=head_dim,
            max_batch_size=self.max_batch_size,
            max_length=inference_config.max_input_len + inference_config.max_output_len,
            block_size=inference_config.block_size,
            kv_max_split_num=kv_max_split_num,
            fd_interm_tensor=fd_inter_tensor,
            dtype=self.dtype,
            device=device,
            enable_streamingllm=inference_config.enable_streamingllm,
            start_token_size=inference_config.start_token_size,
            generated_token_size=inference_config.generated_token_size,
        )

    def _has_running(self) -> bool:
        return not self.running_bb.is_empty()

    def _init_cache(self, model_config):
        self.cache_manager = KVCacheManager(self.inference_config, model_config)

    def get_kvcache(self):
        return self.cache_manager.get_kv_cache()

    def set_spec_dec_mode(self, n_spec_tokens: int):
        self.prefill_bb.set_use_spec_dec(n_spec_tokens)
        self.running_bb.set_use_spec_dec(n_spec_tokens)

    def unset_spec_dec_mode(self):
        self.prefill_bb.reset_use_spec_dec()
        self.running_bb.reset_use_spec_dec()

    def schedule(self):
        """
        The main logic of request handler.
        """
        if self._has_waiting():
            # Try to allocate cache blocks for the sequence using a priority of prompt length.
            for lst in reversed(self.waiting_list):
                if lst:
                    remove_list = []
                    for seq in lst:
                        if seq.input_len > self.inference_config.max_input_len:
                            # If the prompt length is longer than max_input_len, abort the sequence.
                            logger.warning(
                                f"the prompt(Request id = {seq.request_id}) length is longer than max_input_len, abort this sequence."
                            )
                            self.abort_sequence(seq.request_id)
                            remove_list.append(seq)
                            break

                    num_seqs_to_add = min(len(lst), self.max_batch_size - self.running_list.total_seq_num)
                    # for now the recycle logic is not working
                    remove_list.extend(lst[:num_seqs_to_add])
                    self.running_list.extend(lst[:num_seqs_to_add])

                    for seq in remove_list:
                        lst.remove(seq)

        if self.running_list.ready_for_prefill():
            num_seqs_to_add = min(self.running_list.prefill_seq_num, self.prefill_bb.available_batch_size)
            # overwrite the number of sequences to add to 1 if use_spec_dec is enabled
            # TODO (zhaoyuanheng): support speculative decoding for batch size > 1
            if self.prefill_bb.use_spec_dec:
                num_seqs_to_add = 1

            for seq in self.running_list.prefill[:num_seqs_to_add]:
                seq.mark_running()
            # allocate blocks for the prefill batch
            self.prefill_bb.add_seqs(
                self.running_list.prefill[:num_seqs_to_add],
                alloc_block_tables_fn=self.cache_manager.allocate_context_from_block_tables,
            )

            return self.prefill_bb

        if not self.running_bb.is_empty:
            seqs_ids_to_recycle = self.cache_manager.allocate_tokens_from_block_tables(
                self.running_bb.block_tables, self.running_bb.seq_lengths, self.running_bb.current_batch_size
            )
            if seqs_ids_to_recycle:
                seqs_to_recycle = self.running_bb.pop_seqs(seqs_ids_to_recycle)
                for seq in seqs_to_recycle:
                    seq.recycle()
                    self.running_list.remove(seq)
                    self.waiting_list[-1].append(seq)
                    # the recycled sequences are handled with highest priority.

        return self.running_bb

    def allocate_batch_spec_dec(self, batch: BatchBucket, n: int):
        assert batch.use_spec_dec
        if n > 0:
            self.cache_manager.allocate_n_tokens_from_block_tables(
                batch.block_tables, batch.seq_lengths, batch.current_batch_size, n=n
            )

    def add_sequence(self, req: Sequence):
        """
        Add the request to waiting list.
        """
        assert not self._find_sequence(req.request_id), f"Sequence {req.request_id} already exists."
        assert (
            req.input_len <= self.inference_config.max_input_len
        ), f"Sequence {req.request_id} exceeds input length limit"
        self.waiting_list[req.input_len * 3 // (self.inference_config.max_input_len + 1)].append(req)

    def abort_sequence(self, request_id: int):
        """
        Abort the request.
        """
        result = self._find_sequence(request_id)
        if result is not None:
            seq, priority = result
            if seq.status == RequestStatus.WAITING:
                seq.mark_aborted()
                self.waiting_list[priority].remove(seq)
            elif seq.status.is_running():
                self.running_bb.pop_seq_update_batch(seq.request_id, self.cache_manager.free_block_table)
                self.running_list.remove(seq)
            else:
                try:
                    self.done_list.remove(seq)
                except:
                    return
        return

    def _find_sequence(self, request_id: int) -> Sequence:
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

    def update_seq_finished(self, sequence: Sequence, generation_config: GenerationConfig):
        if (
            sequence.output_token_id[-1] == generation_config.eos_token_id
            or sequence.output_len >= generation_config.max_length
        ):
            sequence.mark_finished()

    def update_batch_finished(self, batch: BatchBucket, generation_config: GenerationConfig):
        for seq in batch.seqs_li:
            max_length = generation_config.max_length
            max_new_tokens = generation_config.max_new_tokens
            if max_length is not None:
                max_new_tokens = max_length - seq.input_len
            if seq.output_token_id[-1] == generation_config.eos_token_id or seq.output_len >= max_new_tokens:
                seq.mark_finished()

    def check_unfinished_reqs(self) -> bool:
        return self._has_waiting() or not self.running_list.is_empty()

    def total_requests_in_batch_bucket(self) -> int:
        return self.prefill_bb.current_batch_size + self.running_bb.current_batch_size

    def append_next_tokens(self, sample_tokens: torch.Tensor):
        assert sample_tokens.dim() == 1
        n_elements = sample_tokens.size(0)
        if not self.prefill_bb.is_empty:
            assert (
                self.prefill_bb.current_batch_size == n_elements
            ), f"Incompatible size: {n_elements} tokens to append while prefill batch size {self.prefill_bb.current_batch_size}"
            self.prefill_bb.append_batch_tokens(sample_tokens)
        else:
            assert (
                self.running_bb.current_batch_size == n_elements
            ), f"Incompatible size: {n_elements} tokens to append while running batch size {self.running_bb.current_batch_size}"
            self.running_bb.append_batch_tokens(sample_tokens)

    def update(self):
        """
        Update current running list and done list
        """
        if not self.prefill_bb.is_empty:
            self.running_list.move_prefill_to_decoding(self.prefill_bb.seqs_ids)
            self.running_bb.merge(self.prefill_bb)
            # clear the prefill batch without assigning a free_block_tables_fn
            # since we want to reuse the memory recorded on the block tables
            self.prefill_bb.clear(free_block_tables_fn=None)

        finished_seqs, _ = self.running_bb.pop_finished(self.cache_manager.free_block_table)
        for seq in finished_seqs:
            self.running_list.remove(seq)
        self.done_list.extend(finished_seqs)

        return finished_seqs

    def streamingllm_free_block_tables(self, updated_block_ids: List[int]):
        """
        Free the block that needs to be swapped out.
        """
        self.cache_manager.streamingllm_free_block_tables(updated_block_ids)


class RPCRequestHandler(RequestHandler):
    """
    RPC Version of request handler
    """

    def __init__(self, inference_config: InferenceConfig, model_config: PretrainedConfig) -> None:
        self.inference_config = inference_config
        self.running_list: RunningList = RunningList(inference_config.prefill_ratio)
        self.waiting_list: List[List] = [[], [], []]
        self.done_list: List[Sequence] = []
        self.dtype = inference_config.dtype
        self.max_batch_size = inference_config.max_batch_size

        # initialize cache
        self._init_cache(model_config)

        # initialize batch
        torch.cuda.current_device()
        kv_max_split_num = (
            inference_config.max_input_len + inference_config.max_output_len + inference_config.block_size - 1
        ) // inference_config.block_size
        head_dim = model_config.hidden_size // model_config.num_attention_heads

        # TODO In the continuous batching scenario, the batch size may be greater than max_batch_size,
        # which may cause bugs and this issue should be fixed later.
        self.running_bb = BatchBucket(
            num_heads=model_config.num_attention_heads // inference_config.tp_size,
            head_dim=head_dim,
            max_batch_size=self.max_batch_size,
            max_length=inference_config.max_input_len + inference_config.max_output_len,
            block_size=inference_config.block_size,
            kv_max_split_num=kv_max_split_num,
            fd_interm_tensor=None,
            dtype=self.dtype,
        )
        self.prefill_bb = BatchBucket(
            num_heads=model_config.num_attention_heads // inference_config.tp_size,
            head_dim=head_dim,
            max_batch_size=self.max_batch_size,
            max_length=inference_config.max_input_len + inference_config.max_output_len,
            block_size=inference_config.block_size,
            kv_max_split_num=kv_max_split_num,
            fd_interm_tensor=None,
            dtype=self.dtype,
        )

    def _init_cache(self, model_config):
        self.cache_manager = RPCKVCacheManager(self.inference_config, model_config)
