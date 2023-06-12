from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..generation import generate
from ..lora import LoRAModule
from ..utils import log_probs_from_logits


class Actor(LoRAModule):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self, model: nn.Module, lora_rank: int = 0, lora_train_bias: str = 'none') -> None:
        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.convert_to_lora()

    @staticmethod
    def calc_action_log_probs(output: torch.Tensor,
                              sequences: torch.LongTensor,
                              num_actions: int
                              ) -> torch.Tensor:
        """Calculate action log probs.

        Args:
            output (torch.Tensor): Output tensor of self.forward.
            sequences (torch.LongTensor): Input sequences.
            num_actions (int): Number of actions.

        Returns:
            torch.Tensor: Action log probs.
        """
        logits = output['logits']
        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
        return log_probs[:, -num_actions:]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        return_action_mask: bool = True,
        **kwargs
    ) -> Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:
        # generate sequences
        sequences = generate(self, input_ids, **kwargs)

        # calculate auxiliary tensors
        attention_mask = None
        pad_token_id = kwargs.get('pad_token_id', None)
        if pad_token_id is not None:
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)
        if not return_action_mask:
            return sequences, attention_mask, None
        input_len = input_ids.size(1)
        eos_token_id = kwargs.get('eos_token_id', None)
        if eos_token_id is None:
            action_mask = torch.ones_like(sequences, dtype=torch.bool)
        else:
            # left padding may be applied, only mask action
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
            action_mask = F.pad(action_mask, (1 + input_len, -1), value=True)    # include eos token and input
        action_mask[:, :input_len] = False
        action_mask = action_mask[:, 1:]
        return sequences, attention_mask, action_mask[:, -(sequences.size(1) - input_len):]

    def forward(self,
                sequences: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                **model_kwargs,  # HACK: `generate` method may pass more kwargs
                ) -> torch.Tensor:
        """Returns model output.
        """
        output = self.model(
            sequences,
            attention_mask=attention_mask,
            **model_kwargs
        )
        return output

    def get_base_model(self):
        return self.model
