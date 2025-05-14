"""
Function-based reward verification module.
"""

from typing import Any, Dict, List

import torch


class AGPOReward:
    def __init__(self, reward_fn: callable, **kwargs: List[Dict[str, Any]]):
        self.reward_fn = reward_fn
        self.kwargs = kwargs

    def __call__(
        self,
        input_ids: torch.LongTensor,
        gt_answer: List[torch.Tensor] = None,
        response_idx: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get batch size
        bs = input_ids.size(0)
        num_generations = self.kwargs.get("num_generations")

        # Apply the reward function to the entire batch at once
        reward_infos = [self.reward_fn(input_ids[i], gt_answer=gt_answer[i], response_idx=response_idx[i], **self.kwargs) for i in range(bs)]
        reward_batch = torch.stack([info[0] for info in reward_infos])
        format_acc_batch = torch.stack([info[1] for info in reward_infos])
        ans_acc_batch = torch.stack([info[2] for info in reward_infos])
        seq_len_batch = torch.stack([info[3] for info in reward_infos])

        # calculate mask
        group_reward = reward_batch.view(-1, num_generations)
        reward_std = group_reward.std(dim=1).repeat_interleave(num_generations, dim=0)
        mask = (reward_std == 0) | (ans_acc_batch == 0)

        # process group seq len
        group_seq_len = seq_len_batch.view(-1, num_generations)
        group_mask = mask.view(-1, num_generations)
        masked_seq_len_for_max = group_seq_len.masked_fill(group_mask, -1e6)
        max_group_seq_len = masked_seq_len_for_max.max(dim=1).repeat_interleave(num_generations, dim=0)
        masked_seq_len_for_min = group_seq_len.masked_fill(group_mask, 1e6)
        min_group_seq_len = masked_seq_len_for_min.min(dim=1).repeat_interleave(num_generations, dim=0)

        # correct sample length reward
        len_ratio = (seq_len_batch - min_group_seq_len) / (max_group_seq_len - min_group_seq_len + 1e-6)
        len_rewards = 0.1 * (1 - len_ratio)
        len_rewards = len_rewards.masked_fill(mask, 0.0)
        reward_batch += len_rewards

        rewards = torch.stack([torch.stack([r, f, a]) for r, f, a in zip(reward_batch, format_acc_batch, ans_acc_batch)])
        return rewards
