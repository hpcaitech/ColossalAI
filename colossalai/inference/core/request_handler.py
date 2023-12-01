from typing import List


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
        self.waiting_list: List["Reqseq"] = []
        self.running_list: List["Reqseq"] = []

    def _init_cache(self):
        """
        Initialize the cache manager with cache config.
        """

    def schedule(self):
        """
        The main logic of request handler.
        """

    def add_sequence(self, reqseq: "Reqseq"):
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

    def _find_sequence(self, seq_id: str) -> "Reqseq":
        """
        Find the request by seq_id.
        """

    def check_unfinished_seqs(self) -> bool:
        return self.waiting_list or self.running_list
