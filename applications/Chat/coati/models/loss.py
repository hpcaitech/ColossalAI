import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self):
        super().__init__()
        # NOTE: default ignore_index is -100, which is equal to IGNORE_INDEX in sft_dataset.py
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        log_ratio = log_probs - old_log_probs
        num_steps = (log_ratio.size(1) + chunk_size - 1) // chunk_size
        log_ratio = F.pad(log_ratio, (0, (chunk_size - log_ratio.size(1)) % chunk_size)).view(-1, chunk_size)
        action_mask = F.pad(action_mask, (0, (chunk_size - action_mask.size(1)) % chunk_size)).view(-1, chunk_size)
        chunk_ratio = torch.sum(log_ratio * action_mask, dim=1).exp().view(-1, num_steps)
        surr1 = chunk_ratio * advantages
        surr2 = chunk_ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        # NOTE: loss is likely to be a tensor, not a scalar
        #  requires further reduction to get the final loss
        return loss


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.4) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = 0.5 * torch.max(surr1, surr2)
        # NOTE: loss is likely to be a tensor, not a scalar
        #  requires further reduction to get the final loss
        return loss


class LogSigLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2203.02155
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        return loss


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss
