"""Token-normalised loss helpers for Cosmos 3 training on ColossalAI.

Cosmos 3 predicts vision/action/sound velocities. A pack can contain a different number of supervised tokens on every
rank, so a plain mean per rank is not a global token mean. These helpers keep the numerator and denominator explicit:
the recipe stays in charge of the per-token loss, and only the reduction is shared.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import torch

from colossalai.shardformer.modeling.cosmos_mot import all_reduce_supervised_tokens

__all__ = [
    "velocity_mse",
    "normalise_packed_loss",
    "normalise_vision_token_mean",
    "gather_window_tokens",
    "combine_modality_losses",
]


def velocity_mse(prediction: torch.Tensor, target: torch.Tensor, loss_mask: Optional[torch.Tensor] = None):
    """Return ``(sum, count)`` of a per-element velocity MSE, so the caller can reduce it globally.

    Args:
        prediction (torch.Tensor): Predicted velocity, any shape.
        target (torch.Tensor): Target velocity, same shape as ``prediction``.
        loss_mask (Optional[torch.Tensor], optional): Boolean mask broadcastable to ``prediction``; True means the
            element is supervised. Defaults to None, i.e. every element.

    Returns:
        Tuple[torch.Tensor, int]: Sum of squared errors and the number of supervised elements.

    Raises:
        ValueError: If the shapes disagree or the mask selects no element.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}."
        )
    squared = (prediction.float() - target.float()).square()
    if loss_mask is not None:
        mask = loss_mask.to(device=squared.device, dtype=torch.bool).expand_as(squared)
        squared = squared.masked_select(mask)
    else:
        squared = squared.reshape(-1)
    count = squared.numel()
    if count == 0:
        raise ValueError(
            "The loss mask selected no element. A zero denominator would turn the batch into a NaN; drop the sample "
            "or fix the mask instead."
        )
    return squared.sum(), count


def combine_modality_losses(losses: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    """Sum per-modality ``{"sum": tensor, "count": int}`` entries into one numerator and denominator.

    The numerator stays a tensor so the result is still connected to the autograd graph; the denominator is a Python
    int because it is a reduction bookkeeping value, not a differentiable quantity.

    Args:
        losses (Iterable[Mapping[str, object]]): Entries produced by :func:`velocity_mse` or an equivalent recipe.

    Returns:
        Dict[str, object]: ``{"sum": tensor, "count": int}``. The count is the number of supervised elements.

    Raises:
        ValueError: If no loss is provided or a count is zero.
    """
    total_sum = None
    total_count = 0
    for entry in losses:
        count = int(entry["count"])
        if count <= 0:
            raise ValueError(f"Every modality loss needs a positive count, got {count}.")
        entry_sum = entry["sum"]
        total_sum = entry_sum if total_sum is None else total_sum + entry_sum
        total_count += count
    if total_count == 0 or total_sum is None:
        raise ValueError("No modality loss was provided.")
    return {"sum": total_sum, "count": total_count}


def gather_window_tokens(local_tokens: int, group=None, device: Optional[torch.device] = None) -> int:
    """Sum supervised tokens over the data-parallel group for the current accumulation window.

    Every rank must call this with the same accumulation window. A rank whose pack is empty still participates.
    """
    return all_reduce_supervised_tokens(local_tokens, group=group, device=device)


def normalise_vision_token_mean(
    local_sum,
    local_tokens: int,
    global_tokens: int,
    data_parallel_size: int,
):
    """Scale a local loss sum so ZeRO's data-parallel mean becomes the global token mean.

    ZeRO divides the reduced gradient by the data-parallel size. Passing
    ``local_sum * data_parallel_size / global_tokens`` makes that division reproduce ``sum_all / tokens_all``.

    Args:
        local_sum (torch.Tensor): Sum of the local supervised losses (not a mean). It must still require grad.
        local_tokens (int): Supervised elements in the local microbatch.
        global_tokens (int): Supervised elements over all ranks and accumulated microbatches.
        data_parallel_size (int): Size of the data-parallel gradient group.

    Returns:
        torch.Tensor: The scalar loss to call ``backward`` on. The value stays attached to ``local_sum``.
    """
    if min(local_tokens, global_tokens, data_parallel_size) < 0 or data_parallel_size == 0:
        raise ValueError("Counts must be nonnegative and data_parallel_size positive")
    if local_tokens == 0:
        return local_sum * 0.0
    if global_tokens <= 0 or global_tokens < local_tokens:
        raise ValueError("global_tokens must cover the nonempty local loss")
    return local_sum * (data_parallel_size / global_tokens)


# Compatibility alias. This reduction is a generic element/token mean, not the DROID objective.
normalise_packed_loss = normalise_vision_token_mean
