from typing import Optional

import torch
import torch.nn as nn
from coati.models import BaseModel
from transformers import PretrainedConfig


class RewardModel(BaseModel):
    """
    Reward model class.

    Args:
        model (nn.Module): Critic Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self, pretrained: str = None, config: Optional[PretrainedConfig] = None) -> None:
        super().__init__(pretrained=pretrained, config=config)
        # get last hidden state size with dummy input
        try:
            dummy_outputs = self.model(
                torch.tensor([[1]]).to(self.model.device), attention_mask=torch.tensor([[1]]).to(self.model.device)
            )
            last_hidden_state_size = dummy_outputs["logits"].size(-1)
        except Exception as e:
            raise ValueError(
                f"Please provide a valid pretrained model name or a valid config file for a CasualLM. Caught exception: {e}"
            )

        self.value_head = nn.Linear(last_hidden_state_size, 1)
        self.value_head.weight.data.normal_(mean=0.0, std=1 / (last_hidden_state_size + 1))

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs["logits"]
        sequence_lengths = torch.max(attention_mask * torch.arange(input_ids.size(1), device=input_ids.device), dim=1)[
            0
        ]
        sequence_hidden_states = last_hidden_states[torch.arange(last_hidden_states.size(0)), sequence_lengths].type(
            self.value_head.weight.dtype
        )
        # print("sequence_hidden_states", sequence_hidden_states.size(), sequence_hidden_states.dtype)
        # print("values head weight", self.value_head.weight.size(),self.value_head.weight.dtype)

        values = self.value_head(sequence_hidden_states).squeeze(-1)  # ensure shape is (B,)
        return values
