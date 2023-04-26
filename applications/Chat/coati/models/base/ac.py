from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..generation import generate_with_value
from ..lora import LoRAModule
from ..utils import log_probs_from_logits
from coati.models.base import Actor, Critic



class ActorCritic(nn.Module):
    """
    ActorCritic model class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self, actor:Actor, critic:Critic) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:
        sequences, values = generate_with_value(self.actor.model, self.critic, input_ids, **kwargs)
        
        pad_token_id = kwargs.get('pad_token_id', None)
        if pad_token_id is not None:
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)
            input_mask = input_ids.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)
            # pad input_mask to be as long as attention_mask
            input_mask = F.pad(input_mask, (0, attention_mask.shape[-1]-input_mask.shape[-1], 0, 0), value=0)
            action_mask = attention_mask - input_mask
        
        return sequences, values, attention_mask, action_mask

    def forward(self,
                sequences: torch.LongTensor,
                num_actions: int,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns action log probs
        """
        output = self.model(sequences, attention_mask=attention_mask)
        logits = output['logits']
        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
        return log_probs[:, -num_actions:]

    def get_base_model(self):
        return self.model
