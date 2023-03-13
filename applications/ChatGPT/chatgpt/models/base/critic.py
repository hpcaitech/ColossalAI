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
        lora_train_bias: str = 'none',
        use_action_mask: bool = False,
    ) -> None:

        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.value_head = value_head
        self.use_action_mask = use_action_mask
        self.convert_to_lora()

    def forward(self,
                sequences: torch.LongTensor,
                action_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']

        values = self.value_head(last_hidden_states).squeeze(-1)

        if action_mask is not None and self.use_action_mask:
            num_actions = action_mask.size(1)
            prompt_mask = attention_mask[:, :-num_actions]
            values = values[:, :-num_actions]
            value = masked_mean(values, prompt_mask, dim=1)
            return value

        values = values[:, :-1]
        value = values.mean(dim=1)
        return value
