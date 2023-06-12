from typing import Optional

import torch
import torch.nn as nn

from ..lora import LoRAModule
from ..utils import log_probs_from_logits


class Actor(LoRAModule):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self, model: nn.Module, lora_rank: int = 0, lora_train_bias: str = 'none') -> None:
        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.convert_to_lora()

    @staticmethod
    def calc_action_log_probs(output: torch.Tensor,
                              sequences: torch.LongTensor,
                              num_actions: int
                              ) -> torch.Tensor:
        """Calculate action log probs.

        Args:
            output (torch.Tensor): Output tensor of self.forward.
            sequences (torch.LongTensor): Input sequences.
            num_actions (int): Number of actions.

        Returns:
            torch.Tensor: Action log probs.
        """
        logits = output['logits']
        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
        return log_probs[:, -num_actions:]

    def forward(self,
                sequences: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                **model_kwargs,  # HACK: `generate` method may pass more kwargs
                ) -> torch.Tensor:
        """Returns model output.
        """
        output = self.model(
            sequences,
            attention_mask=attention_mask,
            **model_kwargs
        )
        return output

    def get_base_model(self):
        return self.model
