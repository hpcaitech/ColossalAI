from typing import Optional

import torch.nn as nn
from transformers import BloomConfig, BloomModel

from ..base import Critic


class BLOOMCritic(Critic):
    """
    BLOOM Critic model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (BloomConfig): Model config.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrained: str = None,
        config: Optional[BloomConfig] = None,
        lora_rank: int = 0,
        lora_train_bias: str = "none",
        **kwargs,
    ) -> None:
        if pretrained is not None:
            model = BloomModel.from_pretrained(pretrained)
        elif config is not None:
            model = BloomModel(config)
        else:
            model = BloomModel(BloomConfig())

        value_head = nn.Linear(model.config.hidden_size, 1)
        super().__init__(model, value_head, lora_rank, lora_train_bias, **kwargs)
