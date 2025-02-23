"""
Function-based reward verification module.
"""

from typing import Any, Dict, List

import torch


class VerifiableReward:
    def __init__(self, reward_fn: List[callable], reward_args: List[Dict[str, Any]]):
        self.reward_fn = reward_fn
        self.reward_args = reward_args

    def __call__(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        response_start: List[int] = None,
        response_end: List[int] = None,
        gt_answer: List[str] = None,
    ) -> torch.Tensor:
        # Get batch size
        bs = input_ids.size(0)
        # Initialize reward
        reward = torch.zeros(bs, device=input_ids.device)

        # Loop through reward functions
        for reward_fn in self.reward_fn_list:
            # Apply the reward function to the entire batch at once
            reward_batch = torch.stack(
                [
                    reward_fn(
                        input_ids[i],
                        attention_mask[i],
                        response_start=response_start[i],
                        response_end=response_end[i],
                        gt_answer=gt_answer[i],
                        **self.kwargs,
                    )
                    for i in range(bs)
                ],
                dim=0,
            )

            rewards += reward_batch
        return rewards
