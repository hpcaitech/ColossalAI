from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DrafterOutput:
    """
    Dataclass for drafter model outputs.

    Args:
        speculated_length (int): Speculated length of the output sequence
            It is always less than or equal to spec_num during drafter's speculation process
        logits (torch.FloatTensor): Logits of the output sequence
        next_tokens (torch.Tensor): Next token ids
        past_key_values (Optional[Tuple[Tuple[torch.FloatTensor]]]): Past key values of the output sequence
    """

    speculated_length: int = None
    logits: torch.FloatTensor = None
    next_tokens: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

    def __post_init__(self):
        assert self.speculated_length is not None and self.speculated_length >= 0
        if self.past_key_values is not None:
            assert isinstance(self.past_key_values, tuple), "Past key values should be a tuple"
            assert all([isinstance(past_key_value, tuple) for past_key_value in self.past_key_values])
