"""Same-position diffusion losses, separate from causal next-token loss."""

import torch
import torch.nn.functional as F
from torch import Tensor


def _selected_cross_entropy(logits: Tensor, labels: Tensor):
    if logits.ndim != 3 or not logits.is_floating_point() or 0 in logits.shape:
        raise ValueError("logits must be a nonempty floating tensor with shape (batch, length, vocabulary)")
    if labels.shape != logits.shape[:2] or labels.dtype != torch.long or labels.device != logits.device:
        raise ValueError("labels must be int64 with the logits batch/length shape and device")
    selected = labels != -100
    if ((labels[selected] < 0) | (labels[selected] >= logits.shape[-1])).any():
        raise ValueError("labels must be vocabulary IDs or -100")
    # Keep FP64 for reference/gradcheck; accumulate low-precision losses in FP32.
    values = logits[selected]
    if values.dtype in (torch.float16, torch.bfloat16):
        values = values.float()
    losses = F.cross_entropy(values, labels[selected], reduction="none")
    return selected, losses


def masked_diffusion_loss(
    logits: Tensor,
    labels: Tensor,
    mask_probability: Tensor,
    target_mask: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Compute LLaDA's inverse-probability weighted, per-sequence objective.

    For each sequence: sum(masked-token CE / p) / number_of_eligible_targets.
    The default mean averages these sequence losses across the batch. Pretraining
    uses every valid token as an eligible target; SFT uses the answer tokens.
    EOS padding remains an eligible target if the caller includes it, matching
    the official SFT recipe. Ordinary batch padding should be excluded explicitly.

    Labels are same-position IDs, -100 outside sampled targets: no causal shift.
    Empty sampled sets (or empty target sequences) contribute differentiable zero.
    reduction="none" returns one value per sequence; "sum" supports callers that
    need to normalize across unequal microbatches/ranks. This function performs
    no collectives; averaging unequal microbatch means is not generally correct.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")
    selected, losses = _selected_cross_entropy(logits, labels)
    if target_mask.shape != labels.shape or target_mask.dtype != torch.bool or target_mask.device != logits.device:
        raise ValueError("target_mask must be boolean with the labels shape and device")
    if (selected & ~target_mask).any():
        raise ValueError("supervised labels must belong to target_mask")
    if (
        mask_probability.shape != labels.shape
        or mask_probability.device != logits.device
        or not mask_probability.is_floating_point()
        or not torch.isfinite(mask_probability).all()
        or ((mask_probability <= 0) | (mask_probability > 1)).any()
    ):
        raise ValueError("mask_probability must have the labels shape/device and finite values in (0, 1]")
    rows = selected.nonzero(as_tuple=True)[0]
    weighted = losses / mask_probability.detach()[selected].to(losses.dtype)
    sums = losses.new_zeros(labels.shape[0]).scatter_add(0, rows, weighted)
    sequence_losses = sums / target_mask.sum(dim=1).clamp_min(1).to(sums.dtype)
    if reduction == "none":
        return sequence_losses
    return sequence_losses.sum() if reduction == "sum" else sequence_losses.mean()


def masked_token_cross_entropy(logits: Tensor, labels: Tensor, *, reduction: str = "mean") -> Tensor:
    """Compute dFactory's unweighted same-position masked-token CE.

    This is a separate fine-tuning objective, not LLaDA's 1/p estimator.
    "sum" returns the numerator for explicit token-count normalization across
    microbatches/ranks. An empty sampled set yields differentiable zero.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    _, losses = _selected_cross_entropy(logits, labels)
    numerator = losses.sum()
    return numerator if reduction == "sum" else numerator / max(losses.numel(), 1)
