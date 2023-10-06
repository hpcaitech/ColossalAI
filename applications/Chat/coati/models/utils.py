from typing import Optional, Union

import torch
import torch.nn.functional as F


def _compute_approx_kl(
    log_probs: torch.Tensor, log_probs_base: torch.Tensor, action_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    approx_kl = log_probs_base - log_probs
    if action_mask is not None:
        approx_kl = masked_mean(approx_kl, action_mask, dim=1)
        return approx_kl
    approx_kl = approx_kl.mean(dim=1)
    return approx_kl


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    reward_eps = 5
) -> torch.Tensor:
    '''
    Args:
        log_probs: [batch_size, response_length]
        log_probs_base: [batch_size, response_length]
        action_mask: [batch_size, response_length]
        r: float
    Returns:
        reward: [batch_size, response_length]
    '''
    log_ratio = log_probs - log_probs_base # address numerical instability issue
    kl = -kl_coef * log_ratio * action_mask # _compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    reward = kl
    # print(reward[0]*action_mask[0])
    r_clip = torch.clamp(r, -reward_eps, reward_eps)
    # print(r_clip[0])
    for i in range(action_mask.size(0)):
        assert action_mask[i].sum()>0
        reward[i, :action_mask[i].sum()] += r_clip[i]
        reward[i, action_mask[i].sum():] *= 0
    return reward, ((log_ratio*(log_ratio<10)).exp() - 1 - log_ratio)*action_mask

def _log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean
