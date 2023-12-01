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
    def is_finished(status: "SentenceStatus") -> bool:
        return status in [
            OVERLENGTH,
            COMPLETED,
            LENGTH_CAPPED,
        ]

    @staticmethod
    def is_running(status: "SentenceStatus") -> bool:
        return status == RUNNING

    @staticmethod
    def is_waiting(status: "SentenceStatus") -> bool:
        return status == WAITING


class Sequence:
    """Store the information of a input Sequence.

    Args:
        request_id: The ID of the sequence.
        prompt: The prompt of the sequence.
        token_id: The ID of the sequence.
        block_size: The block size of the sequence.
        sample_params: The sample_params of the sequence.
        block_table_index: The index of this sequence in block_table.
    """

    def __init__(
        self,
        request_id: int,
        prompt: str,
        token_id: int,
        blokc_size: int,
        sample_params: SampleParams,
        block_table_index: int,
    ):
        self.request_id = request_id
        self.input_token_id = token_id
        self.prompt = prompt
        self.blokc_size = blokc_size
        self.output_token_id = []
        self.output = ""
        self.status = SentenceStatus.WAITING
        self.sample_params = sample_params
        self.batch_infer_state = batch_infer_state
        self.block_table_index = block_table_index

    def get_sentence_len(self) -> None:
        return len(self.input_token_id) + len(self.output_token_id)

    def get_input_len(self) -> None:
        return len(self.input_token_id)

    def get_output_len(self) -> None:
        return len(self.output_token_id)

    def check_finish(self) -> bool:
        return SentenceStatus.check_finish(self.status)

    def __repr__(self) -> str:
        return (
            f"Request ID(request_id={self.request_id}, "
            f"prompt={self.prompt}, "
            f"status={self.status.name}, "
            f"sample_params={self.sample_params}, "
            f"logical block number={len(self._logical_blocks)}"
        )


@dataclass
class BatchInferState:
    """
    Information to be passed and used for a batch of sequences.
    """

    sequences_set: Set[Sequence]
    block_table: Dict[int, int]

    @classmethod
    def init_batch(cls, seqs: List[Sequence]) -> BatchInferState:
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
        for seq in self.sequences_set:
            if not seq.check_finish():
                seq.status = RequsetStatus.ABORTED
        self.sequences_set.clear()
        self.block_table.clear()

    def fliter_batch(self) -> None:
        for seq in self.sequences_set:
            if seq.check_finish():
                self.sequences_set.reomve(seq)
                del self.block_table[seq.request_id]

    def add_seqs(self, seqs: List[Sequence]) -> None:
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
