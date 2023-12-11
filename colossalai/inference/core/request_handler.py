from typing import List

from colossalai.inference.kvcache import KVCacheManager


class RequestHandler:
    """
    RequestHandler is the core for handling existing requests and updating current batch.
    During generation process, we call schedule function each iteration to update current batch.

    Args:
       inference_config: Configuration for initialize and manage kv cache.
    """

    def __init__(self, inference_config, block_table) -> None:
        self.inference_config = inference_config
        self._init_cache()
        self.waiting_list: List["Reqseq"] = []
        self.running_list: List[List] = [[], [], []]
        self.batch_handler = BatchHandler(self.inference_config)

    def _init_cache(self):
        """
        Initialize the cache manager with cache config.
        """
        self.cache_manager = KVCacheManager(self.inference_config)

    def _has_waiting(self) -> bool:
        return all(not lst for lst in self.waiting_list)

    def schedule(self):
        """
        The main logic of request handler.
        """
        if self.running_list:
            # remove finished sequences in running_list and free cache blocks.
            for seq in self.running_list:
                if seq.finished:
                    self.cache_manager.free_cache_blocks(seq.block_table)
                    self.running_list.remove(seq)

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
                    if self.cache_manager.can_allocate(seq):
                        # If succeed, add the sequence to running list.
                        self.running_list.append(seq)
                        self.cache_manager.allocate(seq)
                        lst.pop(0)

        self.batch_handler.update_batch(self.running_list)

    def add_sequence(self, req: "Reqseq"):
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
        self.cache_manager.free_cache_blocks(seq.block_table)

    def _find_sequence(self, seq_id: str) -> "Reqseq":
        """
        Find the request by seq_id.
        """
        for seq in self.waiting_list:
            if seq.seq_id == seq_id:
                return seq
        for seq in self.running_list:
            if seq.seq_id == seq_id:
                return seq
        return None

    def check_unfinished_seqs(self) -> bool:
        return self._has_waiting() or self.running_list
