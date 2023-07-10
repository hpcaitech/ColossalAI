from typing import Optional

import torch.nn as nn
from transformers import BloomConfig, BloomForCausalLM, BloomModel

from ..base import RewardModel


class BLOOMRM(RewardModel):
    """
    BLOOM Reward model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (BloomConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: str = None,
                 config: Optional[BloomConfig] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none',
                 freeze_exclude: list = []) -> None:
        if pretrained is not None:
            model = BloomModel.from_pretrained(pretrained)
        elif config is not None:
            model = BloomModel(config)
        else:
            model = BloomModel(BloomConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        value_head = nn.Linear(model.config.hidden_size, 1)
        value_head.weight.data.normal_(mean=0.0, std=1 / (model.config.hidden_size + 1))
        if len(freeze_exclude) > 0:
            for i, layer in enumerate(model.h):
                if i not in freeze_exclude:
                    layer.requires_grad_(False)
        super().__init__(model, value_head, lora_rank, lora_train_bias)
