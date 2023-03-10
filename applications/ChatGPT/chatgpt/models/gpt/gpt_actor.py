from typing import Optional

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from ..base import Actor


class GPTActor(Actor):
    """
    GPT Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (GPT2Config): Model config.
        checkpoint (bool): Enable gradient checkpointing.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[GPT2Config] = None,
                 checkpoint: bool = False) -> None:
        if pretrained is not None:
            model = GPT2LMHeadModel.from_pretrained(pretrained)
        elif config is not None:
            model = GPT2LMHeadModel(config)
        else:
            model = GPT2LMHeadModel(GPT2Config())
        if checkpoint:
            model.gradient_checkpointing_enable()
        super().__init__(model)
