from typing import Optional

import torch.nn as nn
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTModel

from ..base import Critic


class OPTCritic(Critic):
    """
    OPT Critic model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (OPTConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the low-rank approximation.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[OPTConfig] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none',
                 **kwargs) -> None:
        if pretrained is not None:
            model = OPTModel.from_pretrained(pretrained)
        elif config is not None:
            model = OPTModel(config)
        else:
            model = OPTModel(OPTConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        value_head = nn.Linear(model.config.word_embed_proj_dim, 1)
        super().__init__(model, value_head, lora_rank, lora_train_bias, **kwargs)
