"""
Function-based reward verification module.
"""

from typing import Any, Dict, List

import torch


class VerifiableReward:
    def __init__(self, reward_fns: List[callable], **kwargs: List[Dict[str, Any]]):
        self.reward_fns = reward_fns
        self.kwargs = kwargs

    def __call__(
        self,
        input_ids: torch.LongTensor,
        gt_answer: List[torch.Tensor] = None,
        response_idx: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get batch size
        bs = input_ids.size(0)
        # Initialize reward
        rewards = torch.zeros((bs, 3), device=input_ids.device)

        # Loop through reward functions
        for reward_fn in self.reward_fns:
            # Apply the reward function to the entire batch at once
            reward_batch = torch.stack(
                [
                    reward_fn(
                        input_ids[i],
                        gt_answer=gt_answer[i],
                        response_idx=response_idx[i],
                        **self.kwargs,
                    )
                    for i in range(bs)
                ],
                dim=0,
            )

            rewards += reward_batch
        return rewards
