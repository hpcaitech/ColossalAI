from typing import Optional

import torch
import torch.nn as nn
from coati.distributed.utils import masked_mean


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2, skip_threshold: float = 20.0, beta: float = 0.01) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.skip_threshold = skip_threshold
        self.beta = beta

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        per_token_kl: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        skip = False
        if action_mask is None:
            ratio = (log_probs - log_probs.detach()).exp()
        else:
            ratio = ((log_probs - log_probs.detach()) * action_mask).exp()

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2) + self.beta * per_token_kl

        if action_mask is not None:
            loss = masked_mean(loss, action_mask)
        else:
            loss = loss.mean(dim=1)
        if loss_mask is not None:
            loss = loss * loss_mask
        loss = loss.mean()
        return loss, skip, ratio.max()
