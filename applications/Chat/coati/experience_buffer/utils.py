from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from coati.experience_maker.base import Experience


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    values: (1)
    reward: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    reward: torch.Tensor
    kl: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    batch_size = experience.sequences.size(0)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = ("sequences", "action_log_probs", "values", "reward", "kl", "advantages", "attention_mask", "action_mask")
    for key in keys:
        value = getattr(experience, key)
        if isinstance(value, torch.Tensor):
            vals = torch.unbind(value)
        else:
            # None
            vals = [value for _ in range(batch_size)]
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v
    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def _zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem]) -> Experience:
    kwargs = {}
    to_pad_keys = set(("action_log_probs", "action_mask"))
    keys = ("sequences", "action_log_probs", "values", "reward", "kl", "advantages", "attention_mask", "action_mask")
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if key in to_pad_keys:
            batch_data = _zero_pad_sequences(vals)
        else:
            batch_data = torch.stack(vals, dim=0)
        kwargs[key] = batch_data
    return Experience(**kwargs)
