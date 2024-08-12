"""
loss functions
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
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

    def __init__(self, clip_eps: float = 0.2, skip_threshold: float = 20.0) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.skip_threshold = skip_threshold

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        skip = False
        if action_mask is None:
            ratio_ = (log_probs - old_log_probs).exp()
        else:
            ratio_ = ((log_probs - old_log_probs) * action_mask).exp()

        # note that if dropout is disabled (recommanded), ratio will always be 1.
        if ratio_.mean() > self.skip_threshold:
            skip = True

        ratio = ratio_.clamp(0.0, 10.0)
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        if action_mask is not None:
            loss = masked_mean(loss, action_mask)
        else:
            loss = loss.mean(dim=1)
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
        if action_mask is not None:
            loss = torch.sum(torch.max(surr1, surr2) / torch.sum(action_mask) * action_mask)
        else:
            loss = torch.mean(torch.max(surr1, surr2))
        return 0.5 * loss


class DpoLoss(nn.Module):
    """
    Dpo loss
    Details: https://arxiv.org/pdf/2305.18290.pdf

    SimPO loss:
    Details: https://arxiv.org/pdf/2405.14734.pdf
    """

    def __init__(self, beta: float = 0.1, gamma: float = 0.0):
        """
        Args:
            beta: The temperature parameter in the DPO paper.
            gamma: The margin parameter in the SimPO paper.
            length_normalization: Whether to normalize the loss by the length of chosen and rejected responses.
                Refer to the length normalization in the SimPO paper
        """
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        logprob_actor_chosen: torch.Tensor,
        logprob_actor_reject: torch.Tensor,
        logprob_ref_chosen: torch.Tensor,
        logprob_ref_reject: torch.Tensor,
        chosen_mask: torch.Tensor,
        reject_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the DPO/SimPO loss for a batch of policy and reference model log probabilities.

        # adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L328

        Args:
            logprob_actor_chosen: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            logprob_actor_reject: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            logprob_ref_chosen: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            logprob_ref_reject: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            chosen_mask: Mask tensor indicating which responses were chosen. Shape: (batch_size,)
            reject_mask: Mask tensor indicating which responses were rejected. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        logprob_actor_chosen = logprob_actor_chosen * chosen_mask
        logprob_actor_reject = logprob_actor_reject * reject_mask
        if logprob_ref_chosen is not None and logprob_ref_reject is not None:
            logprob_ref_chosen = logprob_ref_chosen * chosen_mask
            logprob_ref_reject = logprob_ref_reject * reject_mask
            if len(logprob_ref_chosen.shape) == 2:
                ref_logratios = logprob_ref_chosen.sum(-1) - logprob_ref_reject.sum(-1)
            else:
                ref_logratios = logprob_ref_chosen - logprob_ref_reject
        else:
            # If no reference model is provided
            ref_logratios = 0.0
        pi_logratios = logprob_actor_chosen.sum(-1) - logprob_actor_reject.sum(-1)
        logits = pi_logratios - ref_logratios - self.gamma / self.beta
        losses = -torch.nn.functional.logsigmoid(self.beta * logits)

        # Calculate rewards for logging
        if logprob_ref_chosen is not None:
            chosen_rewards = self.beta * (logprob_actor_chosen.sum(-1) - logprob_ref_chosen.sum(-1)).detach()
        else:
            chosen_rewards = self.beta * logprob_actor_chosen.sum(-1).detach()
        if logprob_ref_reject is not None:
            rejected_rewards = self.beta * (logprob_actor_reject.sum(-1) - logprob_ref_reject.sum(-1)).detach()
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


class OddsRatioLoss(nn.Module):
    """
    Odds Ratio Loss in ORPO
    Details: https://arxiv.org/pdf/2403.07691
    """

    def forward(
        self,
        chosen_logp: torch.Tensor,
        reject_logp: torch.Tensor,
        chosen_loss_mask: torch.Tensor,
        reject_loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        chosen_logp = chosen_logp.to(dtype=torch.float32)
        reject_logp = reject_logp.to(dtype=torch.float32)
        chosen_odds = chosen_logp - torch.log(-torch.exp(chosen_logp) + 1.0001)
        chosen_odds_masked = torch.sum(chosen_odds * chosen_loss_mask.float()) / torch.sum(chosen_loss_mask)
        reject_odds = reject_logp - torch.log(-torch.exp(reject_logp) + 1.0001)
        reject_odds_masked = torch.sum(reject_odds * reject_loss_mask.float()) / torch.sum(reject_loss_mask)
        log_odds_ratio = chosen_odds_masked - reject_odds_masked
        ratio = torch.log(torch.nn.functional.sigmoid(log_odds_ratio))
        return ratio.to(dtype=torch.bfloat16), log_odds_ratio


class KTOLoss(nn.Module):
    def __init__(self, beta: float = 0.1, desirable_weight: float = 1.0, undesirable_weight: float = 1.0):
        """
        Args:
            beta: The temperature parameter in the KTO paper.
            desirable_weight: The weight for the desirable responses.
            undesirable_weight: The weight for the undesirable
        """
        super().__init__()
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        kl_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        ref_kl_logps: torch.Tensor,
    ):
        """
        Reference:
            https://github.com/huggingface/trl/blob/a2adfb836a90d1e37b1253ab43dace05f1241e04/trl/trainer/kto_trainer.py#L585

        Compute the KTO loss for a batch of policy and reference model log probabilities.
        Args:
            chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            kl_logps: KL divergence of the policy model. Shape: (batch_size,)
            ref_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            ref_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            ref_kl_logps: KL divergence of the reference model. Shape: (batch_size,)
            beta: The temperature parameter in the DPO paper.
            desirable_weight: The weight for the desirable responses.
            undesirable_weight: The weight for the undesirable responses.

        Refer to the KTO paper for details about hyperparameters https://arxiv.org/pdf/2402.01306
        """
        kl = (kl_logps - ref_kl_logps).mean().detach()
        # all gather
        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
        kl = (kl / dist.get_world_size()).clamp(min=0)

        if chosen_logps.shape[0] != 0 and ref_chosen_logps.shape[0] != 0:
            chosen_logratios = chosen_logps - ref_chosen_logps
            chosen_losses = 1 - nn.functional.sigmoid(self.beta * (chosen_logratios - kl))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            chosen_losses = torch.Tensor([]).to(kl_logps.device)
            chosen_rewards = torch.Tensor([]).to(kl_logps.device)

        if rejected_logps.shape[0] != 0 and ref_rejected_logps.shape[0] != 0:
            rejected_logratios = rejected_logps - ref_rejected_logps
            rejected_losses = 1 - nn.functional.sigmoid(self.beta * (kl - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            rejected_losses = torch.Tensor([]).to(kl_logps.device)
            rejected_rewards = torch.Tensor([]).to(kl_logps.device)

        losses = torch.cat((self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0).mean()

        return losses, chosen_rewards, rejected_rewards, kl
