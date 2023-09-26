from typing import Optional

import torch
import torch.nn as nn

from ..lora import LoRAModule


class Actor(LoRAModule):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self, model: nn.Module, lora_rank: int = 0, lora_train_bias: str = "none") -> None:
        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.convert_to_lora()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Returns model output."""
        output = self.model(input_ids, attention_mask=attention_mask, **model_kwargs)
        return output
