import enum
from dataclasses import dataclass
from typing import Any, List

from colossalai.inference.config import DiffusionGenerationConfig
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
    RUNNING = enum.auto()
    ABORTED = enum.auto()

    # completion status
    OVERLENGTH = enum.auto()
    COMPLETED = enum.auto()
    LENGTH_CAPPED = enum.auto()

    # recycle status
    RECYCLED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status in [
            RequestStatus.OVERLENGTH,
            RequestStatus.COMPLETED,
            RequestStatus.LENGTH_CAPPED,
        ]

    @staticmethod
    def is_running(status: "RequestStatus") -> bool:
        return status == RequestStatus.RUNNING

    @staticmethod
    def is_waiting(status: "RequestStatus") -> bool:
        return status == RequestStatus.WAITING


@dataclass
class DiffusionSequence:
    """
    parameters for diffusion
    """

    request_id: int
    prompt: str
    generation_config: DiffusionGenerationConfig


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
        pad_token_id (int): The pad token id for this inference process.
        max_output_len (int): Maximum output length.
        ignore_eos(bool): Whether to ignore the EOS token and continue generating tokens when encountering the EOS token.
        output(str): The output of sequence
    """

    request_id: int
    prompt: str
    input_token_id: List[int]
    block_size: int
    sample_params: Any  # SampleParams needs to be imported later.
    eos_token_id: int
    pad_token_id: int
    max_output_len: int = 256
    # NOTE(caidi) This is a temporary solution. It's better to move the logic to turn on or off the flag in sampling module in future.
    ignore_eos: bool = False
    output: str = None

    def __post_init__(self):
        self.output_token_id = []
        self.status = RequestStatus.WAITING

    @property
    def sentence_len(self) -> int:
        """
        Get length of current sentence.
        """
        return len(self.input_token_id) + len(self.output_token_id)

    @property
    def input_len(self) -> int:
        """
        Get length of input sentence.
        """
        return len(self.input_token_id)

    @property
    def output_len(self) -> int:
        """
        Get length of output sentence.
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
            if (
                self.output_token_id[-1] == self.eos_token_id and not self.ignore_eos
            ) or self.output_len >= self.max_output_len:
                self.status = RequestStatus.COMPLETED
                return True

        return False

    def revoke_finished_status(self) -> None:
        """
        Revoke the finished status of the sequence.
        This is only used by speculative decoding for now.
        """
        if RequestStatus.is_finished(self.status):
            self.status = RequestStatus.RUNNING

    def __hash__(self):
        return hash(self.request_id)

    def mark_running(self) -> None:
        """
        Set status for prefill reqs.
        """
        assert (
            self.status == RequestStatus.WAITING or RequestStatus.RECYCLED
        ), "Sequence is not in WAITTING/RECYCLED STATUS"
        self.status = RequestStatus.RUNNING

    def mark_finished(self) -> None:
        """
        Set status for finished reqs.
        """
        self.status = RequestStatus.COMPLETED

    def mark_aborted(self) -> None:
        """
        Set status for aborted reqs.
        """
        self.status = RequestStatus.ABORTED

    def recycle(self) -> None:
        """
        Recycle a running sequnce to waiitting list
        """
        assert (
            not self.check_finish() and not self.status == RequestStatus.ABORTED
        ), "The running sequence \
        is already done but it still in running list"
        self.status = RequestStatus.RECYCLED

    def __repr__(self) -> str:
        return (
            f"(request_id={self.request_id}, "
            f"prompt={self.prompt},\n"
            f"output_token_id={self.output_token_id},\n"
            f"output={self.output},\n"
            f"status={self.status.name},\n"
            f"sample_params={self.sample_params},\n"
            f"input_len={self.input_len},\n"
            f"output_len={self.output_len})\n"
        )


def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    return [pad] * (max_len - len(x)) + x
