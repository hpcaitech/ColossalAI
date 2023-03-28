from typing import Optional

import torch.nn as nn
from transformers import DebertaV2Config, DebertaV2Model

from ..base import RewardModel


class DebertaRM(RewardModel):
    """
    Deberta Reward model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (DebertaV2Config): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the LO-RA decomposition.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: str = None,
                 config: Optional[DebertaV2Config] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        if pretrained is not None:
            model = DebertaV2Model.from_pretrained(pretrained)
        elif config is not None:
            model = DebertaV2Model(config)
        else:
            model = DebertaV2Model(DebertaV2Config())
        if checkpoint:
            model.gradient_checkpointing_enable()
        value_head = nn.Linear(model.config.hidden_size, 1)
        value_head.weight.data.normal_(mean=0.0, std=1 / (model.config.hidden_size + 1))
        super().__init__(model, value_head, lora_rank, lora_train_bias)
