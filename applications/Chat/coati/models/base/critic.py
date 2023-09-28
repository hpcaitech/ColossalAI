import torch
import torch.nn as nn

from ..lora import LoRAModule


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
        self, model: nn.Module, value_head: nn.Module, lora_rank: int = 0, lora_train_bias: str = "none"
    ) -> None:
        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.value_head = value_head
        self.convert_to_lora()

    def forward(self, sequences: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs["last_hidden_state"]
        # sequence_lengths = torch.max(attention_mask * torch.arange(sequences.size(1), device=sequences.device), dim=1)[
        #     0
        # ]
        sequence_hidden_states = last_hidden_states[torch.arange(last_hidden_states.size(0)), :]
        values = self.value_head(sequence_hidden_states).squeeze(2)  # ensure shape is (B, sequence length)
        return values
