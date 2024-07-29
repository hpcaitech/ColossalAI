"""
Base class for critic and reward model
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig


class BaseModel(nn.Module):
    """
    Actor model base class.

    Args:
        pretrained (str): path to pretrained model.
        config (PretrainedConfig): PretrainedConfig used to initiate the base model.
        **kwargs: all other kwargs as in AutoModel.from_pretrained
    """

    def __init__(self, pretrained: str = None, config: Optional[PretrainedConfig] = None, **kwargs) -> None:
        super().__init__()
        if pretrained is not None:
            if config is not None:
                # initialize with config and load weights from pretrained
                self.model = AutoModel.from_pretrained(pretrained, config=config, **kwargs)
            else:
                # initialize with pretrained
                self.model = AutoModel.from_pretrained(pretrained, **kwargs)
        elif config is not None:
            # initialize with config
            self.model = AutoModel.from_config(config, **kwargs)
        else:
            raise ValueError("Either pretrained or config must be provided.")

        self.config = self.model.config
        # create dummy input to get the size of the last hidden state
        if "use_flash_attention_2" in kwargs:
            self.model = self.model.cuda()
        dummy_input = torch.zeros((1, 1), dtype=torch.long).to(self.model.device)
        out = self.model(dummy_input)
        self.last_hidden_state_size = out.last_hidden_state.shape[-1]
        self.model = self.model.cpu()

    def resize_token_embeddings(self, *args, **kwargs):
        """
        Resize the token embeddings of the model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The resized token embeddings.
        """
        return self.model.resize_token_embeddings(*args, **kwargs)
