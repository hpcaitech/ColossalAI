"""
reward model
"""

from typing import Callable, List, Optional

import torch


class RLVRRewardModel:
    """
    RLVRReward model class. Support varifiable reward.

    Args:
        reward_fn_list List: list of reward functions
        **kwargs: all other kwargs as in reward functions
    """

    def __init__(self, reward_fn_list: List[Callable], **kwargs) -> None:
        self.reward_fn_list = reward_fn_list
        self.kwargs = kwargs

    def __call__(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        response_start: List = None,
        response_end: List = None,
        gt_answer: List = None,
    ) -> torch.Tensor:
        # apply varifiable reward
        bs = input_ids.size(0)
        rewards = torch.zeros(bs, device=input_ids.device)
        for i in range(bs):
            for reward_fn in self.reward_fn_list:
                rewards[i] += reward_fn(
                    input_ids[i],
                    attention_mask[i],
                    response_start=response_start[i],
                    response_end=response_end[i],
                    gt_answer=gt_answer[i],
                    **self.kwargs,
                )
        return rewards

    def to(self, device):
        return self

    def eval(self):
        return self
