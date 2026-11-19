"""Model-independent masked-token corruption and dense block-diffusion inputs.

The block layout follows inclusionAI/dFactory: [noisy sequence, clean sequence].
These helpers do not load models, implement token editing, or change attention
kernels. Dense masks are intended for correctness baselines and short sequences.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class MaskedDiffusionBatch:
    """A corruption draw and the metadata needed for a same-position objective.

    All tensors have shape (batch, length). Labels are -100 outside the sampled
    target positions. target_mask includes *all* eligible targets, including
    unmasked ones, and is therefore the denominator for LLaDA's sequence loss.
    valid_mask describes actual attention tokens, independently of supervision.
    """

    clean_input_ids: Tensor
    input_ids: Tensor
    labels: Tensor
    masked_indices: Tensor
    mask_probability: Tensor
    target_mask: Tensor
    valid_mask: Tensor


def _boolean_mask(mask: Optional[Tensor], like: Tensor, name: str) -> Tensor:
    if mask is None:
        return torch.ones_like(like, dtype=torch.bool)
    if mask.shape != like.shape or mask.dtype != torch.bool or mask.device != like.device:
        raise ValueError(f"{name} must be a boolean tensor with the input shape and device")
    return mask


def sample_masked_tokens(
    input_ids: Tensor,
    mask_token_id: int,
    *,
    target_mask: Optional[Tensor] = None,
    valid_mask: Optional[Tensor] = None,
    probability_range: Tuple[float, float] = (1e-3, 1.0),
    timesteps: Optional[Tensor] = None,
    mask_random: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> MaskedDiffusionBatch:
    """Sample one noise level per sequence and independently mask target tokens.

    p = low + (high - low) * t, where t is uniform on [0, 1). Defaults reproduce
    the LLaDA GUIDELINES.md epsilon schedule. dFactory's published SFT example
    instead uses probability_range=(0.3, 0.8) and an unweighted masked-token loss.

    Supply timesteps (batch,) and mask_random (batch, length) to replay a draw,
    or a device-compatible Generator to save/restore its state. Explicit t may
    equal 1 for boundary tests; mask_random must be in [0, 1). Neither input IDs
    nor masks are modified. Prompt/padding selection is explicit: no token ID
    (including EOS or the mask token itself) is implicitly ignored.
    """
    if input_ids.ndim != 2 or input_ids.dtype != torch.long or 0 in input_ids.shape:
        raise ValueError("input_ids must be a nonempty (batch, length) int64 tensor")
    if not isinstance(mask_token_id, int) or isinstance(mask_token_id, bool) or mask_token_id < 0:
        raise ValueError("mask_token_id must be a nonnegative integer")
    if (input_ids < 0).any():
        raise ValueError("input_ids cannot contain negative token IDs")
    low, high = probability_range
    if not 0 < low <= high <= 1:
        raise ValueError("probability_range must satisfy 0 < low <= high <= 1")
    valid = _boolean_mask(valid_mask, input_ids, "valid_mask")
    target = valid if target_mask is None else _boolean_mask(target_mask, input_ids, "target_mask")
    if (target & ~valid).any():
        raise ValueError("target_mask cannot supervise invalid/padding positions")

    batch, length = input_ids.shape
    if timesteps is None:
        timesteps = torch.rand(batch, device=input_ids.device, generator=generator)
    if (
        timesteps.shape != (batch,)
        or timesteps.device != input_ids.device
        or not timesteps.is_floating_point()
        or not torch.isfinite(timesteps).all()
        or ((timesteps < 0) | (timesteps > 1)).any()
    ):
        raise ValueError("timesteps must be finite floating values in [0, 1] with shape (batch,) on the input device")
    probability = (low + (high - low) * timesteps.detach().float())[:, None].expand(batch, length)
    if mask_random is None:
        mask_random = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    if (
        mask_random.shape != input_ids.shape
        or mask_random.device != input_ids.device
        or not mask_random.is_floating_point()
        or not torch.isfinite(mask_random).all()
        or ((mask_random < 0) | (mask_random >= 1)).any()
    ):
        raise ValueError("mask_random must have the input shape/device and finite floating values in [0, 1)")
    selected = (mask_random < probability) & target
    return MaskedDiffusionBatch(
        clean_input_ids=input_ids.clone(),
        input_ids=torch.where(selected, mask_token_id, input_ids),
        labels=input_ids.masked_fill(~selected, -100),
        masked_indices=selected,
        mask_probability=probability,
        target_mask=target.clone(),
        valid_mask=valid.clone(),
    )


def block_diffusion_attention_mask(
    sequence_length: int,
    block_size: int,
    *,
    device=None,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Return a (batch, 1, 2L, 2L) boolean mask; True means attention is allowed.

    Noisy queries see their own noisy block and strictly earlier clean blocks.
    Clean queries see clean blocks up to and including their own. In particular,
    noisy queries cannot see the clean answer in their current or future block.
    A partial final block is supported. Without padding, batch defaults to one.

    Invalid positions are isolated: valid queries cannot see invalid keys, and
    an invalid query sees only itself to avoid all-masked softmax rows. Packed
    documents and shifted block origins are not supported by this helper.
    """
    for value, name in ((sequence_length, "sequence_length"), (block_size, "block_size")):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if valid_mask is not None:
        if valid_mask.ndim != 2 or valid_mask.shape[1] != sequence_length or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with shape (batch, sequence_length)")
        if valid_mask.shape[0] == 0:
            raise ValueError("valid_mask must have a nonempty batch")
        if device is not None and torch.device(device) != valid_mask.device:
            raise ValueError("device and valid_mask.device must match")
        device = valid_mask.device
    positions = torch.arange(2 * sequence_length, device=device)
    clean = positions >= sequence_length
    blocks = (positions % sequence_length) // block_size
    same_stream = clean[:, None] == clean[None, :]
    same_block = blocks[:, None] == blocks[None, :]
    noisy_to_past_clean = ~clean[:, None] & clean[None, :] & (blocks[:, None] > blocks[None, :])
    clean_to_clean = clean[:, None] & clean[None, :] & (blocks[:, None] >= blocks[None, :])
    allowed = ((same_stream & same_block) | noisy_to_past_clean | clean_to_clean)[None, None]
    if valid_mask is not None:
        valid = torch.cat((valid_mask, valid_mask), dim=1)
        allowed = allowed & valid[:, None, :, None] & valid[:, None, None, :]
        diagonal = torch.eye(2 * sequence_length, device=device, dtype=torch.bool)[None, None]
        allowed = allowed | (diagonal & ~valid[:, None, :, None])
    return allowed


def prepare_block_diffusion_inputs(
    batch: MaskedDiffusionBatch, block_size: int, *, dtype: torch.dtype = torch.float32
) -> Dict[str, Tensor]:
    """Build dense [x_t, x_0] model inputs with duplicated positional indices.

    The model must accept explicit position_ids and a 4D additive attention_mask
    without changing its semantics. Compute loss on logits[:, :L] only. This is
    directly usable by the tested LLaDA2 eager path, not LLaDA-8B's positional API.
    """
    if not dtype.is_floating_point:
        raise ValueError("attention-mask dtype must be floating point")
    batch_size, length = batch.input_ids.shape
    allowed = block_diffusion_attention_mask(length, block_size, valid_mask=batch.valid_mask)
    bias = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(~allowed, float("-inf"))
    positions = torch.arange(length, device=allowed.device).repeat(2)[None].expand(batch_size, -1)
    return {
        "input_ids": torch.cat((batch.input_ids, batch.clean_input_ids), dim=1),
        "position_ids": positions,
        "attention_mask": bias,
    }
