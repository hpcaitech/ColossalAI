from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from coati.models.generation import generate
from coati.models.utils import log_probs_from_logits
from peft import PeftModel
from torch.nn.modules import Module
from transformers import BloomConfig, BloomForCausalLM


class Actor(Module):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, return_action_mask: bool = True, **kwargs
    ) -> Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:
        sequences = generate(self.model, input_ids, **kwargs)
        attention_mask = None
        pad_token_id = kwargs.get("pad_token_id", None)
        if pad_token_id is not None:
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)
        if not return_action_mask:
            return sequences, attention_mask, None
        input_len = input_ids.size(1)
        eos_token_id = kwargs.get("eos_token_id", None)
        if eos_token_id is None:
            action_mask = torch.ones_like(sequences, dtype=torch.bool)
        else:
            # left padding may be applied, only mask action
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
            action_mask = F.pad(action_mask, (1 + input_len, -1), value=True)  # include eos token and input
        action_mask[:, :input_len] = False
        action_mask = action_mask[:, 1:]
        return sequences, attention_mask, action_mask[:, -(sequences.size(1) - input_len) :]

    def forward(
        self, sequences: torch.LongTensor, num_actions: int, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns action log probs"""
        output = self.model(sequences, attention_mask=attention_mask)
        logits = output["logits"]
        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
        return log_probs[:, -num_actions:]

    def get_base_model(self):
        return self.model


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

    def __init__(
        self,
        pretrained: str = None,
        config: Optional[BloomConfig] = None,
        checkpoint: bool = False,
        lora_path: str = None,
    ) -> None:
        if pretrained is not None:
            model = BloomForCausalLM.from_pretrained(pretrained)
        elif config is not None:
            model = BloomForCausalLM(config)
        else:
            model = BloomForCausalLM(BloomConfig())
        if lora_path is not None:
            model = PeftModel.from_pretrained(model, lora_path)
        if checkpoint:
            model.gradient_checkpointing_enable()
        super().__init__(model)

    def print_trainable_parameters(self):
        self.get_base_model().print_trainable_parameters()
