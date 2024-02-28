from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from colossalai.utils import get_current_device

from .struct import DrafterOutput


class Drafter:
    """Container for the Drafter Model (Assistant Model) used in Speculative Decoding.

    Args:
        model (nn.Module): The drafter model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the drafter model.
        max_spec_num (int): The maximum number of tokens to speculate.
        device (torch.device): The device for the drafter model.
    """

    def __init__(
        self, model: nn.Module, tokenizer: PreTrainedTokenizer, max_spec_num: int, device: torch.device = None
    ):
        self._drafter_model = model
        self._tokenizer = tokenizer
        self.max_spec_num = max_spec_num
        self.do_sample = False
        self.sample_fn = None
        self._device = device or get_current_device()
        self._past_key_values = None

    @property
    def past_key_values(self) -> Optional[Tuple[Tuple[torch.FloatTensor]]]:
        return self._past_key_values

    # Debug usage for now
    @property
    def past_key_values_shape(self):
        if self._past_key_values is None:
            return []
        return self._past_key_values[0][0].shape

    def get_model(self) -> nn.Module:
        return self._drafter_model

    def reset_sample_method(self, sample_fn: callable) -> None:
        self.do_sample = True
        self.sample_fn = sample_fn

    def clear_sample_method(self) -> None:
        self.do_sample = False
        self.sample_fn = None

    def reset_max_spec_num(self, n: int) -> None:
        assert isinstance(n, int) and n > 1
        self.max_spec_num = n

    def reset_past_key_values(self, past_key_values: Tuple[Tuple[torch.FloatTensor]] = None) -> None:
        self._past_key_values = past_key_values

    def trim_kv_cache(self, invalid_token_num) -> Tuple[Tuple[torch.FloatTensor]]:
        # Tuple of kv cache tensors: num_layers x 2 x (bsz x num_heads x seq_len x head_dim)
        # Trim the last `invalid_token_num` kv caches
        # The verifier (main model) might reject `invalid_token_num` tokens,
        # and so that we have to trim the invalid tokens for the kv cache of the drafter model.
        assert self._past_key_values is not None
        trimmed_past_key_values = []
        for layer_idx in range(len(self._past_key_values)):
            past_key_value = self._past_key_values[layer_idx]
            trimmed_past_key_values.append(
                (
                    past_key_value[0][:, :, :-invalid_token_num, :],
                    past_key_value[1][:, :, :-invalid_token_num, :],
                )
            )
        self._past_key_values = tuple(trimmed_past_key_values)
        return self._past_key_values

    @torch.inference_mode()
    def speculate(
        self, input_ids: torch.Tensor, n: int, past_key_values: Tuple[Tuple[torch.FloatTensor]] = None
    ) -> DrafterOutput:
        """Generate n tokens using the drafter model.

        Args:
            input_ids (torch.Tensor): Input token ids.
            n (int): Number of tokens to speculate.
            past_key_values (Tuple[Tuple[torch.FloatTensor]]): The past key values of the input sequence.
        """

        assert 0 <= n <= self.max_spec_num, f"Invalid number {n} to speculate"

        # FIXME For compatibility with transformers 4.36.2 (versions before 4.38.0)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if past_key_values is None:
            past_key_values = self._past_key_values

        logits = []
        token_ids = []

        for _ in range(n):
            outputs = self._drafter_model(
                input_ids,
                return_dict=True,
                use_cache=True,
                past_key_values=past_key_values,
            )
            next_token_logits = outputs.logits[:, -1, :]

            # Skip logits_processor for drafter model

            # Sample
            if self.do_sample:
                if self.sample_fn is not None:
                    probs = self.sample_fn(next_token_logits)
                else:
                    probs = nn.functional.softmax(next_token_logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token_ids = torch.argmax(next_token_logits, dim=-1)

            logits.append(next_token_logits)
            token_ids.append(next_token_ids)
            if next_token_ids.item() == self._tokenizer.eos_token_id:
                # TODO support bsz > 1
                break
            input_ids = next_token_ids[:, None]
            past_key_values = outputs.past_key_values

        speculated_length = len(token_ids)  # TODO For now, only support bsz 1
        logits = torch.concat(logits, dim=0)
        token_ids = torch.concat(token_ids, dim=-1)
        # update past_key_values
        self._past_key_values = past_key_values

        out = DrafterOutput(
            speculated_length=speculated_length, logits=logits, next_tokens=token_ids, past_key_values=past_key_values
        )
        return out
