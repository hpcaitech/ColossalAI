from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..generation import generate
from .actor import Actor


class LM(Actor):
    """
    Language model base class.

    Args:
        model (nn.Module): Language Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self, model: nn.Module, lora_rank: int = 0, lora_train_bias: str = 'none') -> None:
        super().__init__(model=model, lora_rank=lora_rank, lora_train_bias=lora_train_bias)

    def forward(self,
                sequences: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns output log probs
        """
        output = self.model(sequences, attention_mask=attention_mask)
        logits = output['logits']
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

