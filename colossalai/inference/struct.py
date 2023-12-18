import enum
from dataclasses import dataclass
from typing import List, Union

import torch
from ordered_set import OrderedSet

from colossalai.logging import get_dist_logger

logger = get_dist_logger(__name__)

"""
The abstraction of request and sequence are defined here.
"""


class RequestStatus(enum.Enum):
    """
    The status of Sentences
    """

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
    def is_finished(status: "RequestStatus") -> bool:
        return status in [
            RequestStatus.OVERLENGTH,
            RequestStatus.COMPLETED,
            RequestStatus.LENGTH_CAPPED,
        ]

    @staticmethod
    def is_running(status: "RequestStatus") -> bool:
        return status in [
            RequestStatus.PREFILL,
            RequestStatus.TOKEN,
        ]

    @staticmethod
    def is_waiting(status: "RequestStatus") -> bool:
        return status == RequestStatus.WAITING


@dataclass
class Sequence:
    """Store information of input sequence.

    Args:
        request_id (int): The ID of input sequence.
        prompt (str): The prompt of input sequence.
        input_token_id (List[int]): The tokens ID of input sequence.
        block_size (int): The block size of input sequence.
        sample_params (SampleParams): The sample_params of input sequence.
        block_table (torch.Tensor): The index of input sequence in block_table.
        eos_token_id (int): The eos token id for this inference process.
        max_output_len (int): Maximum output length.
    """

    request_id: int
    prompt: str
    input_token_id: List[int]
    block_size: int
    sample_params: any  # SampleParams needs to be imported later.
    block_table: torch.Tensor
    eos_token_id: int
    max_output_len: int = 256

    def __post_init__(self):
        self.output_token_id = []
        self.status = RequestStatus.WAITING

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
        Check whether the inference is finished.

        Returns:
            bool: Whether the inference is finished.
        """
        if RequestStatus.is_finished(self.status):
            return True

        if self.output_token_id:
            if self.output_token_id[-1] == self.eos_token_id or len(self.output_token_id) == self.max_output_len:
                self.status = RequestStatus.COMPLETED
                return True

        return False

    def __hash__(self):
        return hash(self.request_id)

    def __repr__(self) -> str:
        return (
            f"Request ID(request_id={self.request_id}, "
            f"prompt={self.prompt}, "
            f"status={self.status.name}, "
            f"sample_params={self.sample_params}"
        )


@dataclass
class BatchInfo:
    """
    Information to be passed and used for a batch of sequences.
    """

    sequences_set: OrderedSet["Sequence"]

    @classmethod
    def init_batch(cls, seqs: List["Sequence"] = None) -> "BatchInfo":
        """
        Initializes inference batches by input sentence list.

        Args:
            seqs (List["Sequence"]): List of input sequence.
        """

        sequences_set = OrderedSet()

        if seqs is not None:
            if not isinstance(seqs, list):
                seqs = [seqs]
            for seq in seqs:
                if seq in sequences_set:
                    logger.warning(f"The sequence(request_id {seq.request_id}) is already in sequences_set.")
                    continue

                sequences_set.add(seq)

        return cls(sequences_set=sequences_set)

    def get_block_table_tensor(self):
        tesnor_list = []
        for seq in self.sequences_set:
            block_table = seq.block_table
            assert block_table, f"The sequence(request_id {seq.request_id}) has not initialized the block_table."
            tesnor_list.append(seq.block_table)
        return torch.concat(tesnor_list)

    def clear_batch(self) -> None:
        """
        Clear sequence set and block table.
        """
        for seq in self.sequences_set:
            if not seq.check_finish():
                seq.status = RequestStatus.ABORTED
        self.sequences_set.clear()

    def fliter_batch(self) -> List["Sequence"]:
        """
        Remove completed sentences from a batch.

        Returns:
            List["Sequence"]: List of finished sequences.
        """
        finish_seqs = []
        for seq in self.sequences_set:
            if seq.check_finish():
                finish_seqs.append(seq)
        for finish_seq in finish_seqs:
            self.sequences_set.discard(finish_seq)
        return finish_seqs

    def abort_seq(self, seq: "Sequence") -> "Sequence":
        """
        Remove sequence from the batch.
        """
        if not seq.check_finish():
            seq.status = RequestStatus.ABORTED
        self.sequences_set.discard(seq)
        return seq

    def add_seqs(self, seqs: List["Sequence"]) -> None:
        """
        Add new sequence to batch

        Args:
            seqs (List["Sequence"]): The list of new sequences.
        """

        if not isinstance(seqs, list):
            seqs = [seqs]

        for seq in seqs:
            if seq in self.sequences_set:
                logger.warning(f"The sequence(request_id {seq.request_id}) is already in sequences_set.")
                continue
            self.sequences_set.add(seq)

    def is_empty(self) -> None:
        """
        Check whether sequences_set is empty.
        """
        return not self.sequences_set

    def update_batch_tokens(self, tokens: Union[List[int], List[List[int]]]) -> None:
        """
        Add an output token for each sentence in the batch.

        Args:
            tokens (List[int]): A batch of tokens
        """

        assert self.get_batch_size() == len(tokens), "The number of tokens does not match batch_size."

        for seq, token in zip(self.sequences_set, tokens):
            if not isinstance(token, list):
                if not isinstance(token, int):
                    raise TypeError(f"The token type must be List[int] or int, but get {type(token)}.")
                token = [token]
            seq.output_token_id += token
            seq.check_finish()

    def get_batch_size(self) -> int:
        """
        Get batch_size of this batch
        """
        return len(self.sequences_set)
