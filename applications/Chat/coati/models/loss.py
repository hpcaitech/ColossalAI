from typing import Optional

import torch
import torch.nn as nn

from .utils import masked_mean


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
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        skip = False
        # log_probs = log_probs + 1e-6
        # old_log_probs = old_log_probs + 1e-6
        ratio_ = ((log_probs - old_log_probs) * action_mask).exp()
        # if masked_mean(ratio_, action_mask).max()>50.0:
        #     skip = True
        ratio = ratio_.clamp(0.0, 10.0)
        # advantages = advantages.clamp(-0.7, 0.7)
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask)
        loss = loss.mean()
        return loss, skip, ratio_.max()


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        advantage: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        returns = advantage + old_values
        values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2) / torch.sum(action_mask)
        loss = torch.sum(loss * action_mask)
        return 0.5 * loss


class DpoLoss(nn.Module):
    """
    Dpo loss
    Details: https://arxiv.org/pdf/2305.18290.pdf
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        logprob_actor_chosen: torch.Tensor,
        logprob_actor_reject: torch.Tensor,
        logprob_ref_chosen: torch.Tensor,
        logprob_ref_reject: torch.Tensor,
    ):
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        # adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L328

        Args:
            logprob_actor_chosen: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            logprob_actor_reject: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            logprob_ref_chosen: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            logprob_ref_reject: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        if logprob_ref_chosen is not None and logprob_ref_reject is not None:
            ref_logratios = logprob_ref_chosen - logprob_ref_reject
        else:
            ref_logratios = torch.zeros_like(logprob_actor_chosen)

        pi_logratios = logprob_actor_chosen - logprob_actor_reject
        logits = pi_logratios.sum(-1) - ref_logratios.sum(-1)
        losses = -torch.nn.functional.logsigmoid(self.beta * logits)
        if logprob_ref_chosen is not None:
            chosen_rewards = self.beta * (logprob_actor_chosen - logprob_ref_chosen).sum(-1).detach()
        else:
            chosen_rewards = self.beta * logprob_actor_chosen.sum(-1).detach()
        if logprob_ref_reject is not None:
            rejected_rewards = self.beta * (logprob_actor_reject - logprob_ref_reject).sum(-1).detach()
        else:
            rejected_rewards = self.beta * logprob_actor_reject.sum(-1).detach()

        return losses, chosen_rewards, rejected_rewards


class LogSigLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2203.02155
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        return -torch.nn.functional.logsigmoid(chosen_reward - reject_reward).mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss
