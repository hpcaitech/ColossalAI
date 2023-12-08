import enum
from dataclasses import dataclass
from typing import Dict, List

from ordered_set import OrderedSet


class RequsetStatus(enum.Enum):
    """The status of Sentences"""

    # running status
    WAITING = enum.auto()
    PREFILL = enum.auto()
    TOKEN = enum.auto()
    ABORTED = enum.auto()

    # completion status
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
        return (
            status
            == status
            in [
                RequsetStatus.PREFILL,
                RequsetStatus.TOKEN,
            ]
        )

    @staticmethod
    def is_waiting(status: "RequsetStatus") -> bool:
        return status == RequsetStatus.WAITING


class Sequence:
    """Store information of input sequence.

    Args:
        request_id: The ID of input sequence.
        prompt: The prompt of input sequence.
        input_token_id: The tokens ID of input sequence.
        block_size: The block size of input sequence.
        sample_params: The sample_params of input sequence.
        block_table_index: The index of input sequence in block_table.
    """

    def __init__(
        self,
        request_id: int,
        prompt: str,
        input_token_id: List[int],
        block_size: int,
        sample_params,  # SampleParams needs to be imported later.
        block_table_index: int,
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.input_token_id = input_token_id
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
        )


@dataclass
class BatchHandler:
    """
    Information to be passed and used for a batch of sequences.
    """

    sequences_set: OrderedSet[Sequence]
    block_table: Dict[int, int]

    @classmethod
    def init_batch(cls, seqs: List[Sequence]) -> "BatchHandler":
        """
        Initializes inference batches by input sentence list.

        Args:
            seqs (List[Sequence]): List of input sequence.
        """
        if not isinstance(seqs, list):
            seqs = [seqs]

        sequences_set = OrderedSet()
        block_table = {}
        for seq in seqs:
            if seq in sequences_set:
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

    def fliter_batch(self) -> List[Sequence]:
        """
        Remove completed sentences from a batch.
        """
        finish_seqs = []
        for seq in self.sequences_set:
            if seq.check_finish():
                finish_seqs.append(seq)
                self.sequences_set.discard(seq)
                del self.block_table[seq.request_id]
        return finish_seqs

    def add_seqs(self, seqs: List[Sequence]) -> None:
        """
        Add new sequence to batch

        Args:
            seqs (List[Sequence]): The list of new sequences.
        """

        if not isinstance(seqs, list):
            seqs = [seqs]

        for seq in seqs:
            if seq in self.sequences_set:
                assert (
                    seq.request_id in self.block_table
                ), "The sequence has been added to sequences_set, but it has not been added to block_table."
                continue
            assert (
                seq.request_id not in self.block_table
            ), "The sequence has not been added to sequences_set, but it has already been added to block_table."
            self.sequences_set.add(seq)
            self.block_table[seq.request_id] = seq.block_table_index

    def is_empty(self):
        assert len(self.sequences_set) == len(
            self.block_table
        ), "The length of sequences_set does not match the length of block_table."
        return not self.sequences_set
