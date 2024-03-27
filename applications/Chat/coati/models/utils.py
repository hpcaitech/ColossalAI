import torch
import torch.nn.functional as F


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """
    # FIXME: this fn is not used for now
    raise NotImplementedError


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
