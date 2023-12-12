from typing import List

from .inference_struct import BatchHandler, Sequence


class RequestHandler:
    """
    RequestHandler is the core for handling existing requests and updating current batch.
    During generation process, we call schedule function each iteration to update current batch.

    Args:
        cache_config: Configuration for initialize and manage kv cache.
    """

    def __init__(self, cache_config) -> None:
        self.cache_config = cache_config
        self._init_cache()
        self.waiting_list: List["Sequence"] = []
        self.running_list: List["Sequence"] = []
        self.batch = BatchHandler.init_batch([])

    def _init_cache(self):
        """
        Initialize the cache manager with cache config.
        """

    def schedule(self):
        """
        The main logic of request handler.
        """
        # The code below is only used for testing engine and will be modified.
        if self.waiting_list:
            self.running_list = self.waiting_list
        self.batch.add_seqs(self.running_list)
        return self.batch

    def add_sequence(self, reqseq: "Sequence"):
        """
        Add the request to waiting list.
        """
        self.waiting_list.append(reqseq)

    def abort_sequence(self, seq_id: str):
        """
        Abort the request. #TODO :implement this
        """
        self._find_sequence(seq_id)
        return

    def _find_sequence(self, seq_id: str) -> "Sequence":
        """
        Find the request by seq_id.
        """

    def check_unfinished_seqs(self) -> bool:
        return self.waiting_list or self.running_list

    def update(self):
        """
        Update the waiting list and running list.
        """

        # The code below is only used for testing engine and will be modified.
        self.waiting_list = []
        self.running_list = []
        finished_sequences = list(self.batch.sequences_set)

        self.batch.clear_batch()
        return finished_sequences
