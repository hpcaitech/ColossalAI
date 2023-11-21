from typing import Optional

import torch
from coati.models import BaseModel
from transformers import PretrainedConfig


class Actor(BaseModel):
    """
    Actor model base class.

    Args:
        pretrained (str): path to pretrained model.
        config (PretrainedConfig): PretrainedConfig used to initiate the base model.
    """

    def __init__(self, pretrained: str = None, config: Optional[PretrainedConfig] = None) -> None:
        super().__init__(pretrained=pretrained, config=config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Returns model output."""
        output = self.model(input_ids, attention_mask=attention_mask, **model_kwargs)
        return output
