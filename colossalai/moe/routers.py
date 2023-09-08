import math
from abc import ABC
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

from colossalai.context import MOE_CONTEXT
from colossalai.nn.layer.moe._operation import moe_cumsum
from colossalai.utils import get_current_device


class MoeRouter(nn.Module, ABC):
    """Base class for all MoE routers.
    Args:
        k_value (int): The value of top_k.
        capacity_factor_train (float): Capacity factor in routing of training.
        capacity_factor_eval (float): Capacity factor in routing of evaluation.
        min_capacity (int): The minimum number of the capacity of each expert.
        noisy_func (:class:`typing.Callable`, optional): Noisy function used in logits.
        drop_tks (bool, optional): Whether drops tokens in evaluation
    """

    def __init__(self,
                 k_value: int,
                 capacity_factor_train: float,
                 capacity_factor_eval: float,
                 min_capacity: int,
                 noisy_func: Callable = None,
                 drop_tks: bool = True):
        super().__init__()
        self.k_value = k_value
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.min_capacity = min_capacity
        self.noisy_func = noisy_func
        self.drop_tks = drop_tks
        self._aux_loss = None
        self._z_loss = None

    def get_capacity(self, logits_shape):
        capacity_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        capacity = math.floor(self.k_value * capacity_factor * logits_shape[-2] / logits_shape[-1])
        capacity += capacity % 2
        capacity = max(capacity, self.min_capacity)
        assert capacity > 0
        return capacity

    def set_aux_loss(self, logits: torch.Tensor, cmask: torch.Tensor, num_experts: int) -> None:
        assert self._aux_loss is None
        me = torch.mean(logits, dim=0)
        ce = torch.mean(cmask.float(), dim=0)
        aux_loss = num_experts * torch.sum(me * ce)
        self._aux_loss = aux_loss

    def set_z_loss(self, router_logits: torch.Tensor):
        assert self._z_loss is None
        n, _ = router_logits.shape
        log_z = torch.logsumexp(router_logits, axis=-1)
        z_loss = log_z**2
        z_loss = torch.sum(z_loss, dtype=torch.float32) / n
        self._z_loss = z_loss

    def pop_router_loss(self) -> torch.Tensor:
        assert self._aux_loss is not None
        MOE_CONTEXT.add_loss(self._aux_loss, self._z_loss)
        self._aux_loss = None
        self._z_loss = None


class Top1Router(MoeRouter):
    """Top1 router that returns the dispatch mask [s, e, c] and combine weight [s, e, c]
    for routing usage. More detailed function can be found in the paper about Switch Transformer
    of Google.
    Args:
        capacity_factor_train (float, optional): Capacity factor in routing of training.
        capacity_factor_eval (float, optional): Capacity factor in routing of evaluation.
        min_capacity (int, optional): The minimum number of the capacity of each expert.
        select_policy (str, optional): The policy about tokens selection.
        noisy_func (:class:`typing.Callable`, optional): Noisy function used in logits.
        drop_tks (bool, optional): Whether drops tokens in evaluation
    """

    def __init__(self,
                 capacity_factor_train: float = 1.25,
                 capacity_factor_eval: float = 2.0,
                 min_capacity: int = 4,
                 select_policy: str = "first",
                 noisy_func: Callable = None,
                 drop_tks: bool = True):
        super().__init__(k_value=1,
                         capacity_factor_train=capacity_factor_train,
                         capacity_factor_eval=capacity_factor_eval,
                         min_capacity=min_capacity,
                         noisy_func=noisy_func,
                         drop_tks=drop_tks)
        self.select_policy = select_policy
        assert select_policy in {"first", "random"}
        if select_policy == "random":
            self.uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=get_current_device()),
                                                               high=torch.tensor(1.0,
                                                                                 device=get_current_device())).rsample

    def forward(self, inputs: torch.Tensor, use_kernel: bool = False, ep_group: Optional[ProcessGroup] = None):

        if self.noisy_func is not None and self.training:
            inputs = self.noisy_func(inputs)

        assert inputs.dtype == torch.float
        logits = F.softmax(inputs, dim=-1)
        num_experts = logits.size(-1)
        capacity = self.get_capacity(logits.shape)

        top1_idx = torch.argmax(inputs, dim=-1)
        mask = F.one_hot(top1_idx, num_classes=num_experts).to(torch.int32)

        # caculate router loss
        self.set_aux_loss(logits, mask, num_experts)
        self.set_z_loss(inputs)
        self.pop_router_loss()

        if not self.training and not self.drop_tks and ep_group is not None:
            max_num = torch.max(torch.sum(mask, dim=0))
            dist.all_reduce(max_num, op=dist.ReduceOp.MAX, group=ep_group)
            capacity = max_num.item()

        if self.select_policy == "random":
            rand_mask = mask * self.uniform(mask.shape)
            _, dispatch_idx = torch.topk(rand_mask, k=capacity, dim=0)
            mask = mask * torch.zeros_like(mask).scatter_(0, dispatch_idx, 1)
            ranks = moe_cumsum(mask)
        elif self.select_policy == "first":
            ranks = moe_cumsum(mask)
            mask = mask * torch.lt(ranks, capacity)
        else:
            raise NotImplementedError("Not support such select policy yet.")

        ranks = torch.sum(mask * ranks, dim=-1)

        if use_kernel:
            mask = torch.sum(mask, dim=-1)
            mask = torch.stack([mask], dim=0).to(torch.int32)
            dest_idx = torch.stack([top1_idx * capacity + ranks], dim=0).to(torch.int32)
            return logits, mask, dest_idx, num_experts * capacity
        else:
            ranks = F.one_hot(ranks, num_classes=capacity)
            weight = mask * logits.type_as(inputs)
            combine_weights = weight.unsqueeze(2) * ranks.unsqueeze(1)
            sec_mask = combine_weights.bool()
            return combine_weights, sec_mask


class Top2Router(MoeRouter):
    """Top2 router that returns the dispatch mask [s, e, c] and combine weight [s, e, c]
    for routing usage. More detailed function can be found in the paper about ViT-MoE.
    Args:
        capacity_factor_train (float, optional): Capacity factor in routing of training.
        capacity_factor_eval (float, optional): Capacity factor in routing of evaluation.
        min_capacity (int, optional): The minimum number of the capacity of each expert
        noisy_func (:class:`typing.Callable`, optional): Noisy function used in logits.
        drop_tks (bool, optional): Whether drops tokens in evaluation.
    """

    def __init__(self,
                 capacity_factor_train: float = 1.25,
                 capacity_factor_eval: float = 2.0,
                 min_capacity: int = 4,
                 noisy_func: Callable = None,
                 drop_tks: bool = True):
        super().__init__(k_value=2,
                         capacity_factor_train=capacity_factor_train,
                         capacity_factor_eval=capacity_factor_eval,
                         min_capacity=min_capacity,
                         noisy_func=noisy_func,
                         drop_tks=drop_tks)

    def forward(self, inputs: torch.Tensor, use_kernel: bool = False, ep_group: Optional[ProcessGroup] = None):
        # inputs: [s, h]
        if self.noisy_func is not None and self.training:
            inputs = self.noisy_func(inputs)

        assert inputs.dtype == torch.float
        logits = F.softmax(inputs, dim=-1)    # logits: [s, e]
        num_experts = logits.size(-1)
        capacity = self.get_capacity(logits.shape)

        top1_idx = torch.argmax(logits, dim=-1)
        mask1 = F.one_hot(top1_idx, num_classes=num_experts).to(torch.int32)
        logits_except1 = logits.masked_fill(mask1.bool(), float("-inf"))
        top2_idx = torch.argmax(logits_except1, dim=-1)
        mask2 = F.one_hot(top2_idx, num_classes=num_experts).to(torch.int32)

        cmask = (mask1 + mask2)    # loss: [s, e]
        cmask = cmask.float() / 2.0    # div 2 to normalize it to 1

        # caculate loss
        self.set_aux_loss(logits, cmask, num_experts)
        self.set_z_loss(inputs)
        self.pop_router_loss()

        if not self.training and not self.drop_tks and ep_group is not None:
            max_num = torch.max(torch.sum(cmask, dim=0))
            dist.all_reduce(max_num, op=dist.ReduceOp.MAX, group=ep_group)
            capacity = max_num.item()

        rank1 = moe_cumsum(mask1)    # rank1: [s, e]
        rank2 = moe_cumsum(mask2)
        rank2 += torch.sum(mask1, dim=-2, keepdim=True)

        mask1 *= torch.lt(rank1, capacity)
        mask2 *= torch.lt(rank2, capacity)

        rank1 = torch.sum(mask1 * rank1, dim=-1)
        rank2 = torch.sum(mask2 * rank2, dim=-1)

        if use_kernel:
            mask1 = torch.sum(mask1, dim=-1)
            mask2 = torch.sum(mask2, dim=-1)

            mask = torch.stack([mask1, mask2], dim=0).to(torch.int32)
            dest_idx = torch.stack([top1_idx * capacity + rank1, top2_idx * capacity + rank2], dim=0).to(torch.int32)

            return logits, mask, dest_idx, num_experts * capacity
        else:
            weight1 = mask1 * logits.type_as(inputs)
            weight2 = mask2 * logits.type_as(inputs)
            rank1_sc = F.one_hot(rank1, num_classes=capacity)
            rank2_sc = F.one_hot(rank2, num_classes=capacity)

            cb_weight1 = weight1.unsqueeze(2) * rank1_sc.unsqueeze(1)
            cb_weight2 = weight2.unsqueeze(2) * rank2_sc.unsqueeze(1)
            cb_weight = cb_weight1 + cb_weight2
            sec_mask = cb_weight.bool()

            return cb_weight, sec_mask


def get_router_cls(top_k: int) -> MoeRouter:
    if top_k == 1:
        router_cls = Top1Router
    elif top_k == 2:
        router_cls = Top2Router
    else:
        raise NotImplementedError("top_k > 2 is not supported yet")
    return router_cls
