"""
reward model
"""

from typing import Optional

import torch
import torch.nn as nn
from coati.models import BaseModel
from transformers import PretrainedConfig


class RewardModel(BaseModel):
    """
    Reward model class.

    Args:
        pretrained str: huggingface or local model path
        config: PretrainedConfig object
        **kwargs: all other kwargs as in AutoModel.from_pretrained
    """

    def __init__(self, pretrained: str = None, config: Optional[PretrainedConfig] = None, **kwargs) -> None:
        super().__init__(pretrained=pretrained, config=config, **kwargs)
        self.value_head = nn.Linear(self.last_hidden_state_size, 1)
        self.value_head.weight.data.normal_(mean=0.0, std=1 / (self.last_hidden_state_size + 1))

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask=attention_mask)

        last_hidden_states = outputs["last_hidden_state"]
        sequence_lengths = torch.max(attention_mask * torch.arange(input_ids.size(1), device=input_ids.device), dim=1)[
            0
        ]
        sequence_hidden_states = last_hidden_states[torch.arange(last_hidden_states.size(0)), sequence_lengths].type(
            self.value_head.weight.dtype
        )
        values = self.value_head(sequence_hidden_states).squeeze(-1)  # Ensure shape is (B,)
        return values

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()
