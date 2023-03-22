from typing import Optional

import torch.nn as nn
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaModel

from ..base import Critic


class RoBERTaCritic(Critic):
    """
    RoBERTa Critic model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (RoBERTa Config): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the low-rank approximation.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[RobertaConfig] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none',
                 **kwargs) -> None:
        if pretrained is not None:
            model = RobertaModel.from_pretrained(pretrained)
        elif config is not None:
            model = RobertaModel(config)
        else:
            model = RobertaModel(RobertaConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        value_head = nn.Linear(model.config.hidden_size, 1)
        super().__init__(model, value_head, lora_rank, lora_train_bias, **kwargs)
