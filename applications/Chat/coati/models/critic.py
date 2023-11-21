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

    def __init__(self, pretrained: str = None, config: Optional[PretrainedConfig] = None) -> None:
        super().__init__(pretrained=pretrained, config=config)
        # get last hidden state size with dummy input
        try:
            dummy_outputs = self.model(
                torch.tensor([[1]]).to(self.model.device), attention_mask=torch.tensor([[1]]).to(self.model.device)
            )
            last_hidden_state_size = dummy_outputs["last_hidden_state"].size(-1)
        except Exception as e:
            raise ValueError(
                f"Please provide a valid pretrained model name or a valid config file for a CasualLM. Caught exception: {e}"
            )

        self.value_head = nn.Linear(last_hidden_state_size, 1)

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs["last_hidden_state"]
        sequence_hidden_states = last_hidden_states[torch.arange(last_hidden_states.size(0)), :]
        values = self.value_head(sequence_hidden_states).squeeze(2)  # ensure shape is (B, sequence length)
        return values
