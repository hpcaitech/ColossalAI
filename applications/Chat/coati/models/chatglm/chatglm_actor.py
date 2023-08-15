from typing import Optional

import torch

from ..base import Actor
from .configuration_chatglm import ChatGLMConfig
from .modeling_chatglm import ChatGLMForConditionalGeneration


class ChatGLMActor(Actor):
    """
    ChatGLM Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (BloomConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: str = None,
                 config: Optional[ChatGLMConfig] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        if pretrained is not None:
            model = ChatGLMForConditionalGeneration.from_pretrained(pretrained)
        elif config is not None:
            model = ChatGLMForConditionalGeneration(config)
        else:
            model = ChatGLMForConditionalGeneration(ChatGLMConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        if lora_rank != 0 or lora_train_bias != 'none':
            import warnings
            warnings.warn('LoRA is not supported for ChatGLM yet.')
            lora_rank = 0
            lora_train_bias = 'none'
        super().__init__(model, lora_rank, lora_train_bias)
