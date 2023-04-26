from typing import Optional

import torch
import torch.nn as nn

from ..lora import LoRAModule
from ..utils import masked_mean


class Critic(LoRAModule):
    """
    Critic model base class.

    Args:
        model (nn.Module): Critic model.
        value_head (nn.Module): Value head to get value.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        model: nn.Module,
        value_head: nn.Module,
        lora_rank: int = 0,
        lora_train_bias: str = 'none'
    ) -> None:

        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.value_head = value_head
        self.convert_to_lora()

    def forward(self,
                sequences: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # we have added a <eos> token at the end of each sequence
        # so we can use the last hidden state to be the input of the marking linear layer to get the value score 
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']
        value_prob = last_hidden_states[:, -1]
        value = self.value_head(value_prob).squeeze(1) # ensure shape is (B)
        return value
