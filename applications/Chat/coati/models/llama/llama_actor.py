from typing import Optional

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from ..base import Actor


class LlamaActor(Actor):
    """
    Llama Actor model.

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
                 lora_train_bias: str = 'none') -> None:

        if pretrained is not None:
            model = LlamaForCausalLM.from_pretrained(pretrained)
        elif config is not None:
            model = LlamaForCausalLM(config)
        else:
            model = LlamaForCausalLM(LlamaConfig())

        if checkpoint:
            model.gradient_checkpointing_enable()

        super().__init__(model, lora_rank, lora_train_bias)
