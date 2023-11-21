from typing import Optional

import torch.nn as nn
from transformers import AutoModelForCausalLM, PretrainedConfig


class BaseModel(nn.Module):
    """
    Actor model base class.

    Args:
        pretrained (str): path to pretrained model.
        config (PretrainedConfig): PretrainedConfig used to initiate the base model.
    """

    def __init__(self, pretrained: str = None, config: Optional[PretrainedConfig] = None) -> None:
        super().__init__()
        if pretrained is not None:
            if config is not None:
                # initialize with config and load weights from pretrained
                self.model = AutoModelForCausalLM.from_pretrained(pretrained, config=config)
            else:
                # initialize with pretrained
                self.model = AutoModelForCausalLM.from_pretrained(pretrained)
        elif config is not None:
            # initialize with config
            self.model = AutoModelForCausalLM(config)
        else:
            raise ValueError("Either pretrained or config must be provided.")
