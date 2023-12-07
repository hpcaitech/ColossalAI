import enum
from dataclasses import dataclass
from typing import Dict, List, Set


class RequsetStatus(enum.Enum):
    """The status of Sentences"""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    ABORTED = enum.auto()
    OVERLENGTH = enum.auto()
    COMPLETED = enum.auto()
    LENGTH_CAPPED = enum.auto()

    @staticmethod
    def is_finished(status: "RequsetStatus") -> bool:
        return status in [
            RequsetStatus.OVERLENGTH,
            RequsetStatus.COMPLETED,
            RequsetStatus.LENGTH_CAPPED,
        ]

    @staticmethod
    def is_running(status: "RequsetStatus") -> bool:
        return status == RequsetStatus.RUNNING

    @staticmethod
    def is_waiting(status: "RequsetStatus") -> bool:
        return status == RequsetStatus.WAITING


class Sequence:
    """Store information of input sequence.

    Args:
        request_id: The ID of input sequence.
        prompt: The prompt of input sequence.
        token_id: The tokens ID of input sequence.
        block_size: The block size of input sequence.
        sample_params: The sample_params of input sequence.
        block_table_index: The index of input sequence in block_table.
    """

    def __init__(
        self,
        request_id: int,
        prompt: str,
        token_id: List[int],
        block_size: int,
        sample_params,  # SampleParams needs to be imported later.
        block_table_index: int,
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.input_token_id = token_id
        self.blokc_size = block_size
        self.sample_params = sample_params
        self.output_token_id = []
        self.status = RequsetStatus.WAITING
        self.block_table_index = block_table_index

    def get_sentence_len(self) -> None:
        """
        Get length of current sentence.
        """
        return len(self.input_token_id) + len(self.output_token_id)

    def get_input_len(self) -> None:
        """
        Get length of input sentence.
        """
        return len(self.input_token_id)

    def get_output_len(self) -> None:
        """
        Get output length of current sentence.
        """
        return len(self.output_token_id)

    def check_finish(self) -> bool:
        """
        Check whether inference is over.
        """
        return RequsetStatus.is_finished(self.status)

    def __repr__(self) -> str:
        return (
            f"Request ID(request_id={self.request_id}, "
            f"prompt={self.prompt}, "
            f"status={self.status.name}, "
            f"sample_params={self.sample_params}, "
            f"logical block number={len(self._logical_blocks)}"
        )


@dataclass
class BatchHandler:
    """
    Information to be passed and used for a batch of sequences.
    """

    sequences_set: Set[Sequence]
    block_table: Dict[int, int]

    @classmethod
    def init_batch(cls, seqs: List[Sequence]) -> "BatchHandler":
        """
        Initializes inference batches by input sentence list.

        Args:
            seqs (List[Sequence]): List of input sequence.
        """
        sequences_set = set()
        block_table = {}
        for seq in seqs:
            if seq in sequences_set:
                print("The sequence is already in sequences_set.")
                assert (
                    seq.request_id in block_table
                ), "The sequence has been added to sequences_set, but it has not been added to block_table."
                continue
            assert (
                seq.request_id not in block_table
            ), "The sequence has not been added to sequences_set, but it is already in block_table."

            sequences_set.add(seq)
            block_table[seq.request_id] = seq.block_table_index

        return cls(sequences_set=sequences_set, block_table=block_table)

    def clear_batch(self) -> None:
        """
        Clear sequence set and block table.
        """
        for seq in self.sequences_set:
            if not seq.check_finish():
                seq.status = RequsetStatus.ABORTED
        self.sequences_set.clear()
        self.block_table.clear()

    def fliter_batch(self) -> None:
        """
        Remove completed sentences from a batch.
        """
        for seq in self.sequences_set:
            if seq.check_finish():
                self.sequences_set.reomve(seq)
                del self.block_table[seq.request_id]

    def add_seqs(self, seqs: List[Sequence]) -> None:
        """
        Add new sequence to batch

        Args:
            seqs (List[Sequence]): The list of new sequences.
        """
        for seq in seqs:
            if seq in self.sequences_set:
                print("The sequence is already in sequences_set.")
                assert (
                    seq.request_id in self.block_table
                ), "The sequence has been added to sequences_set, but it has not been added to block_table."
                continue
            assert (
                seq.request_id not in self.block_table
            ), "The sequence has not been added to sequences_set, but it is already in block_table."
            self.sequences_set.add(seq)
            self.block_table[seq.request_id] = seq.block_table_index
