import json
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F


def get_model_numel(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    reward_eps=5,
) -> torch.Tensor:
    """
    Args:
        log_probs: [batch_size, response_length]
        log_probs_base: [batch_size, response_length]
        action_mask: [batch_size, response_length]
        r: float
    Returns:
        reward: [batch_size, response_length]
    """
    log_ratio = log_probs - log_probs_base  # address numerical instability issue
    kl = -kl_coef * log_ratio * action_mask
    reward = kl
    r_clip = torch.clamp(r, -reward_eps, reward_eps)
    for i in range(action_mask.size(0)):
        assert action_mask[i].sum() > 0
        reward[i, : action_mask[i].sum()] += r_clip[i]
        reward[i, action_mask[i].sum() :] *= 0
    return reward, ((log_ratio * (log_ratio < 10)).exp() - 1 - log_ratio) * action_mask


def _log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the log probabilities from logits for the given labels.

    Args:
        logits (torch.Tensor): The input logits.
        labels (torch.Tensor): The target labels.

    Returns:
        torch.Tensor: The log probabilities corresponding to the labels.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def calc_action_log_probs(logits: torch.Tensor, sequences: torch.LongTensor, num_actions: int) -> torch.Tensor:
    """Calculate action log probs.

    Args:
        output (torch.Tensor): Output tensor of Actor.forward.logits.
        sequences (torch.LongTensor): Input sequences.
        num_actions (int): Number of actions.

    Returns:
        torch.Tensor: Action log probs.
    """
    log_probs = _log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
    return log_probs[:, -num_actions:]


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Compute the masked mean of a tensor along a specified dimension.

    Args:
        tensor (torch.Tensor): The input tensor.
        mask (torch.Tensor): The mask tensor with the same shape as the input tensor.
        dim (int, optional): The dimension along which to compute the mean. Default is 1.

    Returns:
        torch.Tensor: The masked mean tensor.

    """
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean


def calc_masked_log_probs(
    logits: torch.Tensor, sequences: torch.LongTensor, mask: torch.Tensor, length_normalization: bool = False
) -> torch.Tensor:
    """
    Calculate the masked log probabilities for a given sequence of logits.

    Args:
        logits (torch.Tensor): The input logits tensor of shape (batch_size, sequence_length, vocab_size).
        sequences (torch.LongTensor): The input sequence tensor of shape (batch_size, sequence_length).
        mask (torch.Tensor): The mask tensor of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: The masked log probabilities tensor of shape (batch_size, sequence_length - 1).
    """
    # logits are probabilities of the next token, so we shift them to the left by one
    log_probs = _log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])

    if not length_normalization:
        return log_probs * mask
    else:
        return log_probs * mask / (mask.sum(dim=-1, keepdim=True) + 0.01)


def load_json(file_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Load file in JSON format
    """
    with open(file=file_path, mode="r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(data: Dict[str, Any], file_path: Union[str, os.PathLike]) -> None:
    """
    Save as JSON format
    """
    with open(file=file_path, mode="w", encoding="utf-8") as fp:
        json.dump(data, fp=fp, ensure_ascii=False, indent=4)


def disable_dropout(model: torch.nn.Module):
    """
    Disables dropout in a PyTorch model. This is used in PPO Training

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        None
    """
    if model is not None:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
