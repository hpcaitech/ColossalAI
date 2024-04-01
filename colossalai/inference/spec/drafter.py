from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from colossalai.utils import get_current_device

from .struct import DrafterOutput, GlideInput


class Drafter:
    """Container for the Drafter Model (Assistant Model) used in Speculative Decoding.

    Args:
        model (nn.Module): The drafter model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the drafter model.
        device (torch.device): The device for the drafter model.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
    ):
        self._tokenizer = tokenizer
        self._device = device or get_current_device()
        self._dtype = dtype
        self._drafter_model = model.to(self._device)
        self._drafter_model = model.to(self._dtype)
        self._drafter_model.eval()

    def get_model(self) -> nn.Module:
        return self._drafter_model

    @staticmethod
    def trim_kv_cache(
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]], invalid_token_num: int
    ) -> Tuple[Tuple[torch.FloatTensor]]:
        """Trim the last `invalid_token_num` kv caches.

        past_key_values (Tuple[Tuple[torch.FloatTensor]]): The past key values with shape
            num_layers x 2 x (bsz x num_heads x seq_len x head_dim)
        invalid_token_num (int): The number of invalid tokens to trim.
        """
        if past_key_values is None or invalid_token_num < 1:
            return past_key_values

        trimmed_past_key_values = []
        for layer_idx in range(len(past_key_values)):
            past_key_value = past_key_values[layer_idx]
            trimmed_past_key_values.append(
                (
                    past_key_value[0][:, :, :-invalid_token_num, :],
                    past_key_value[1][:, :, :-invalid_token_num, :],
                )
            )
        past_key_values = tuple(trimmed_past_key_values)
        return past_key_values

    @torch.inference_mode()
    def speculate(
        self,
        input_ids: torch.Tensor,
        n_spec_tokens: int,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        glide_input: Optional[GlideInput] = None,
    ) -> DrafterOutput:
        """Generate n_spec_tokens tokens using the drafter model.

        Args:
            input_ids (torch.Tensor): Input token ids.
            n_spec_tokens (int): Number of tokens to speculate.
            past_key_values (Tuple[Tuple[torch.FloatTensor]]): The past key values of the input sequence.
            glide_input (Optional[GlideInput]): The packed input for glimpsing kv caches of the main model,
                when using the glide model as a drafter.
        """
        assert n_spec_tokens >= 1, f"Invalid number {n_spec_tokens} to speculate"

        # For compatibility with transformers of versions before 4.38.0
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        logits = []
        token_ids = []

        kwargs = {"return_dict": True, "use_cache": True}
        if glide_input:
            # required only when using glide model
            kwargs["glide_input"] = glide_input

        for _ in range(n_spec_tokens):
            # update past key values
            kwargs["past_key_values"] = past_key_values

            outputs = self._drafter_model(input_ids, **kwargs)
            next_token_logits = outputs.logits[:, -1, :]

            # NOTE Only use greedy search for speculating.
            #      As the drafter model usually has only a few layers with few parameters,
            #      introducing sampling will make the speculation unstable and lead to worse performance.
            next_token_ids = torch.argmax(next_token_logits, dim=-1)

            logits.append(next_token_logits)
            token_ids.append(next_token_ids)
            if next_token_ids.item() == self._tokenizer.eos_token_id:
                # TODO(yuanheng-zhao) support bsz > 1
                break
            input_ids = next_token_ids[:, None]
            past_key_values = outputs.past_key_values

        speculated_length = len(token_ids)  # For now, only support bsz 1
        logits = torch.concat(logits, dim=0)
        token_ids = torch.concat(token_ids, dim=-1)

        out = DrafterOutput(
            speculated_length=speculated_length, logits=logits, next_tokens=token_ids, past_key_values=past_key_values
        )
        return out
