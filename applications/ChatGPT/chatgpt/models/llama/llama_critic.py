from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from ..base import Critic


class LlamaCritic(Critic):
    """
    Llama Critic model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (LlamaConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[LlamaConfig] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none',
                 **kwargs) -> None:

        if pretrained is not None:
            model = LlamaForCausalLM.from_pretrained(pretrained)
        elif config is not None:
            model = LlamaForCausalLM(config)
        else:
            model = LlamaForCausalLM(LlamaConfig())

        if checkpoint:
            model.gradient_checkpointing_enable()

        value_head = nn.Linear(model.config.hidden_size, 1)

        super().__init__(model, value_head, lora_rank, lora_train_bias, **kwargs)
