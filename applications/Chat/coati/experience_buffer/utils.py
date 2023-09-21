from dataclasses import dataclass
from typing import List

import torch
from coati.experience_maker.base import Experience


@dataclass
class BufferItem:
    """BufferItem is an item of `Experience` data.

    Shapes of each tensor:
        sequences: (S)
        attention_mask: (S)
        action_mask: (A)
        step_mask: (N)
        action_log_probs: (A)
        values: (N)
        returns: (N)
        advantages: (N)

    """

    sequences: torch.Tensor
    attention_mask: torch.LongTensor
    action_mask: torch.BoolTensor
    step_mask: torch.BoolTensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    batch_size = experience.sequences.size(0)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "attention_mask",
        "action_mask",
        "step_mask",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
    )
    for key in keys:
        value = getattr(experience, key)
        assert isinstance(value, torch.Tensor)
        vals = torch.unbind(value)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v
    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def make_experience_batch(items: List[BufferItem]) -> Experience:
    kwargs = {}
    keys = (
        "sequences",
        "attention_mask",
        "action_mask",
        "step_mask",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        batch_data = torch.stack(vals, dim=0)
        kwargs[key] = batch_data
    return Experience(**kwargs)
