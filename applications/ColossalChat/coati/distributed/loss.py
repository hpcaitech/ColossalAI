from typing import Optional

import torch
import torch.nn as nn
from coati.distributed.utils import masked_mean, masked_sum


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        beta: float = 0.01,
        loss_variation: str = "sample_level",
        adv: str = "GRPO",
    ) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.beta = beta
        self.loss_variation = loss_variation
        assert loss_variation in ["sample_level", "token_level"], f"Unsupported loss variation: {loss_variation}"
        self.adv = adv

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        per_token_kl: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        total_effective_tokens_in_batch: torch.Tensor = None,
    ) -> torch.Tensor:
        if action_mask is None:
            ratio = (log_probs - old_log_probs.detach()).exp()
        else:
            ratio = ((log_probs - old_log_probs.detach()) * action_mask).exp()

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages
        if self.beta == 0:
            # skip kl term if kl coefficient is zero
            per_token_kl = 0.0
        loss = -torch.min(surr1, surr2) + self.beta * per_token_kl

        if self.loss_variation == "sample_level":
            if action_mask is not None:
                loss = masked_mean(loss, action_mask)
            else:
                loss = loss.mean(dim=1)
            if loss_mask is not None:
                loss = loss * loss_mask
            loss = loss.mean()
        elif self.loss_variation == "token_level":
            if action_mask is not None:
                loss = masked_sum(loss, action_mask)
            else:
                loss = loss.sum(dim=1)
            if loss_mask is not None:
                loss = loss * loss_mask
            loss = loss.sum() / (total_effective_tokens_in_batch + 1e-8)
        else:
            raise ValueError(f"Unsupported loss variation: {self.loss_variation}")

        return loss, ratio.max()
