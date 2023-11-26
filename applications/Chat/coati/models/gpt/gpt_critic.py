from typing import Optional

import torch.nn as nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from ..base import Critic


class GPTCritic(Critic):
    """
    GPT Critic model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (GPT2Config): Model config.
        lora_rank (int): Rank of the LO-RA decomposition.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrained: Optional[str] = None,
        config: Optional[GPT2Config] = None,
        lora_rank: int = 0,
        lora_train_bias: str = "none",
        **kwargs,
    ) -> None:
        if pretrained is not None:
            model = GPT2Model.from_pretrained(pretrained)
        elif config is not None:
            model = GPT2Model(config)
        else:
            model = GPT2Model(GPT2Config())

        value_head = nn.Linear(model.config.n_embd, 1)
        super().__init__(model, value_head, lora_rank, lora_train_bias, **kwargs)
