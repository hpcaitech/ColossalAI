"""
Critic model
"""

from typing import Optional

import torch
import torch.nn as nn
from coati.models import BaseModel
from transformers import PretrainedConfig


class Critic(BaseModel):
    """
    Critic model class.

    Args:
        pretrained (str): path to pretrained model.
        config (PretrainedConfig): PretrainedConfig used to initiate the base model.
    """

    def __init__(self, pretrained: str = None, config: Optional[PretrainedConfig] = None, **kwargs) -> None:
        super().__init__(pretrained=pretrained, config=config, **kwargs)
        # et last hidden state size with dummy input
        self.value_head = nn.Linear(self.last_hidden_state_size, 1)

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs["last_hidden_state"]
        sequence_hidden_states = last_hidden_states[torch.arange(last_hidden_states.size(0)), :].type(
            self.value_head.weight.dtype
        )
        values = self.value_head(sequence_hidden_states).squeeze(-1)  # ensure shape is (B, sequence length)
        return values

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()
