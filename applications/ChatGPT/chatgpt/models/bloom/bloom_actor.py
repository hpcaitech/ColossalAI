from typing import Optional

import torch
from transformers import BloomConfig, BloomForCausalLM, BloomModel

from ..base import Actor


class BLOOMActor(Actor):
    """
    BLOOM Actor model.

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
                 lora_train_bias: str = 'none') -> None:
        if pretrained is not None:
            model = BloomForCausalLM.from_pretrained(pretrained)
        elif config is not None:
            model = BloomForCausalLM(config)
        else:
            model = BloomForCausalLM(BloomConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        super().__init__(model, lora_rank, lora_train_bias)
