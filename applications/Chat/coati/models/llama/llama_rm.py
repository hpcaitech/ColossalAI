from typing import Optional

import torch.nn as nn
from transformers import LlamaConfig, LlamaModel

from ..base import RewardModel


class LlamaRM(RewardModel):
    """
    Llama Reward model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (LlamaConfig): Model config.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrained: Optional[str] = None,
        config: Optional[LlamaConfig] = None,
        lora_rank: int = 0,
        lora_train_bias: str = "none",
    ) -> None:
        if pretrained is not None:
            model = LlamaModel.from_pretrained(pretrained)
        elif config is not None:
            model = LlamaModel(config)
        else:
            model = LlamaModel(LlamaConfig())

        value_head = nn.Linear(model.config.hidden_size, 1)
        value_head.weight.data.normal_(mean=0.0, std=1 / (model.config.hidden_size + 1))

        super().__init__(model, value_head, lora_rank, lora_train_bias)
