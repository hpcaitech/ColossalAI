import math
from abc import ABC
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

from colossalai.accelerator import get_accelerator
from colossalai.moe._operation import moe_cumsum
from colossalai.moe.manager import MOE_MANAGER


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

    def __init__(
        self,
        k_value: int,
        capacity_factor_train: float,
        capacity_factor_eval: float,
        min_capacity: int,
        noisy_func: Optional[Callable] = None,
        drop_tks: bool = True,
        use_kernel: bool = False,
    ):
        super().__init__()
        self.k_value = k_value
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.min_capacity = min_capacity
        self.noisy_func = noisy_func
        self.drop_tks = drop_tks
        self._aux_loss = None
        self._z_loss = None
        self.use_kernel = use_kernel

    def get_capacity(self, num_tokens, num_experts, ep_group=None):
        if ep_group is not None:
            num_tokens_tensor = torch.tensor(num_tokens, device=get_accelerator().get_current_device())
            dist.all_reduce(num_tokens_tensor, group=ep_group)
            num_tokens = num_tokens_tensor.item() // dist.get_world_size(ep_group)
        capacity_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        capacity = math.floor(self.k_value * capacity_factor * num_tokens / num_experts)
        capacity += capacity % 2
        capacity = max(capacity, self.min_capacity)
        assert capacity > 0
        return int(capacity)

    def set_aux_loss(self, router_probs: torch.Tensor, expert_indices: torch.Tensor, num_experts: int) -> None:
        """Computes auxiliary load balancing loss as in Switch Transformer.

        See Switch Transformer (https://arxiv.org/abs/2101.03961). This function
        implements the loss function presented in equations (4) - (6). It aims to
        penalize those cases where the routing between experts is unbalanced.

        Args:
            router_probs: Probability assigned to each expert per token. Shape:
                <float32>[num_groups, tokens_per_group, num_experts].
            expert_indices: <int>[num_groups, tokens_per_group, num_selected_experts]
                indices identifying the top num_selected_experts for a given token.
        """
        assert self._aux_loss is None
        if router_probs.dim() == expert_indices.dim() == 2:
            router_probs = router_probs.unsqueeze(0)
            expert_indices = expert_indices.unsqueeze(0)
        assert (
            router_probs.dim() == expert_indices.dim() == 3
        ), "router_probs must be 3D tensor and expert_indices must be 4D tensor"

        # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
        expert_mask = F.one_hot(expert_indices, num_experts)
        # For a given token, determine if it was routed to a given expert.
        # Shape: [num_groups, tokens_per_group, num_experts]
        expert_mask = expert_mask.max(dim=-2)[0]

        tokens_per_group_and_expert = torch.mean(expert_mask.float(), dim=-2)
        router_prob_per_group_and_expert = torch.mean(router_probs.float(), dim=-2)
        aux_loss = num_experts**2 * torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert)
        self._aux_loss = aux_loss

    def set_z_loss(self, router_logits: torch.Tensor):
        """Compute router z-loss.

        The router z-loss was introduced in Designing Effective Sparse Expert Models
        (https://arxiv.org/abs/2202.08906). It encourages router logits to remain
        small in an effort to improve stability.

        Args:
            router_logits: <float>[num_groups, tokens_per_group, num_experts] router logits.
        """
        assert self._z_loss is None
        if router_logits.dim() == 2:
            router_logits = router_logits.unsqueeze(0)
        assert router_logits.dim() == 3, "router_logits must be 3D tensor"
        num_groups, tokens_per_group, _ = router_logits.shape
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = torch.sum(log_z**2, dtype=torch.float32) / (num_groups * tokens_per_group)
        self._z_loss = z_loss

    def pop_router_loss(self) -> torch.Tensor:
        assert self._aux_loss is not None
        MOE_MANAGER.add_loss(self._aux_loss, self._z_loss)
        self._aux_loss = None
        self._z_loss = None


class Top1Router(MoeRouter):
    """Top1 router that returns the dispatch mask (batch_size * seq_len, num_experts, capacity)
    and combine weight (batch_size * seq_len, num_experts, capacity) for routing usage. More detailed
    function can be found in the paper about Switch Transformer of Google.

    Args:
        capacity_factor_train (float, optional): Capacity factor in routing of training.
        capacity_factor_eval (float, optional): Capacity factor in routing of evaluation.
        min_capacity (int, optional): The minimum number of the capacity of each expert.
        select_policy (str, optional): The policy about tokens selection.
        noisy_func (:class:`typing.Callable`, optional): Noisy function used in logits.
        drop_tks (bool, optional): Whether drops tokens in evaluation
    """

    def __init__(
        self,
        capacity_factor_train: float = 1.25,
        capacity_factor_eval: float = 2.0,
        min_capacity: int = 4,
        select_policy: str = "first",
        noisy_func: Optional[Callable] = None,
        drop_tks: bool = True,
    ):
        super().__init__(
            k_value=1,
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_eval,
            min_capacity=min_capacity,
            noisy_func=noisy_func,
            drop_tks=drop_tks,
        )
        self.select_policy = select_policy
        assert select_policy in {"first", "random"}
        if select_policy == "random":
            self.uniform = torch.distributions.uniform.Uniform(
                low=torch.tensor(0.0, device=get_accelerator().get_current_device()),
                high=torch.tensor(1.0, device=get_accelerator().get_current_device()),
            ).rsample

    def forward(
        self,
        inputs: torch.Tensor,
        use_kernel: bool = False,
        ep_group: Optional[ProcessGroup] = None,
        use_loss: bool = False,
        use_norm: bool = False,
    ) -> Tuple:
        """
        Args:
            inputs (torch.Tensor): The input tensor of shape (batch_size * seq_len, num_experts).

        Returns:
            1. use_kernel is False:
                The combine weight tensor of shape (batch_size * seq_len, num_experts, capacity).
                The dispatch mask tensor of shape (batch_size * seq_len, num_experts, capacity).
            2. use_kernel is True:
                ...
        """
        if self.noisy_func is not None and self.training:
            inputs = self.noisy_func(inputs)

        assert inputs.dtype == torch.float
        probs = F.softmax(inputs, dim=-1)
        num_experts = probs.size(-1)
        num_tokens = inputs.size(0)
        capacity = self.get_capacity(num_tokens, num_experts, ep_group)

        top1_idx = torch.argmax(inputs, dim=-1)
        mask = F.one_hot(top1_idx, num_classes=num_experts).to(torch.int32)

        # calculate router loss
        self.set_aux_loss(probs, top1_idx.unsqueeze(-1), num_experts)
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
            ranks = moe_cumsum(mask, use_kernel=self.use_kernel)
        elif self.select_policy == "first":
            ranks = moe_cumsum(mask, use_kernel=self.use_kernel)
            mask = mask * torch.lt(ranks, capacity)
        else:
            raise NotImplementedError("Not support such select policy yet.")

        ranks = torch.sum(mask * ranks, dim=-1)
        used_capacity = mask.sum(dim=0)

        if use_kernel:
            mask = torch.sum(mask, dim=-1)
            mask = torch.stack([mask], dim=0).to(torch.int32)
            dest_idx = torch.stack([top1_idx * capacity + ranks], dim=0).to(torch.int32)
            return used_capacity, probs, mask, dest_idx, num_experts * capacity
        else:
            ranks = F.one_hot(ranks, num_classes=capacity)
            weight = mask * probs.type_as(inputs)
            combine_weights = weight.unsqueeze(2) * ranks.unsqueeze(1)
            sec_mask = combine_weights.bool()
            return used_capacity, combine_weights, sec_mask, probs


class Top2Router(MoeRouter):
    """Top2 router that returns the dispatch mask (batch_size * seq_len, num_experts, capacity)
    and combine weight (batch_size * seq_len, num_experts, capacity) for routing usage. More detailed
    function can be found in the paper about ViT-MoE.

    Args:
        capacity_factor_train (float, optional): Capacity factor in routing of training.
        capacity_factor_eval (float, optional): Capacity factor in routing of evaluation.
        min_capacity (int, optional): The minimum number of the capacity of each expert
        noisy_func (:class:`typing.Callable`, optional): Noisy function used in logits.
        drop_tks (bool, optional): Whether drops tokens in evaluation.
    """

    def __init__(
        self,
        capacity_factor_train: float = 1.25,
        capacity_factor_eval: float = 2.0,
        min_capacity: int = 4,
        noisy_func: Optional[Callable] = None,
        drop_tks: bool = True,
    ):
        super().__init__(
            k_value=2,
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_eval,
            min_capacity=min_capacity,
            noisy_func=noisy_func,
            drop_tks=drop_tks,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        use_kernel: bool = False,
        ep_group: Optional[ProcessGroup] = None,
        use_norm: bool = False,
        use_loss: bool = True,
    ) -> Tuple:
        """
        Args:
            inputs (torch.Tensor): The input tensor of shape (batch_size * seq_len, num_experts).

        Returns:
            1. use_kernel is False:
                The combine weight tensor of shape (batch_size * seq_len, num_experts, capacity).
                The dispatch mask tensor of shape (batch_size * seq_len, num_experts, capacity).
            2. use_kernel is True:
                ...
        """
        if self.noisy_func is not None and self.training:
            inputs = self.noisy_func(inputs)

        assert inputs.dtype == torch.float
        probs = F.softmax(inputs, dim=-1)
        if use_norm:
            routing_weights, _ = torch.topk(probs, 2, dim=-1)
            probs = probs / routing_weights.sum(dim=-1, keepdim=True)

        num_experts = probs.size(-1)
        num_tokens = inputs.size(0)
        capacity = self.get_capacity(num_tokens, num_experts, ep_group)

        top1_idx = torch.argmax(probs, dim=-1)
        mask1 = F.one_hot(top1_idx, num_classes=num_experts).to(torch.int32)
        logits_except1 = probs.masked_fill(mask1.bool(), float("-inf"))
        top2_idx = torch.argmax(logits_except1, dim=-1)
        mask2 = F.one_hot(top2_idx, num_classes=num_experts).to(torch.int32)

        cmask = mask1 + mask2  # loss: [s, e]
        cmask = cmask.float() / 2.0  # div 2 to normalize it to 1

        # calculate loss
        if use_loss:
            expert_indices = torch.stack([top1_idx, top2_idx], dim=-1)
            self.set_aux_loss(probs, expert_indices, num_experts)
            self.set_z_loss(inputs)
            self.pop_router_loss()

        if not self.training and not self.drop_tks and ep_group is not None:
            max_num = torch.max(torch.sum(cmask, dim=0))
            dist.all_reduce(max_num, op=dist.ReduceOp.MAX, group=ep_group)
            capacity = max_num.item()

        rank1 = moe_cumsum(mask1, use_kernel=self.use_kernel)  # rank1: [s, e]
        rank2 = moe_cumsum(mask2, use_kernel=self.use_kernel)
        rank2 += torch.sum(mask1, dim=-2, keepdim=True)

        mask1 *= torch.lt(rank1, capacity)
        mask2 *= torch.lt(rank2, capacity)
        used_capacity = mask1.sum(dim=0) + mask2.sum(dim=0)

        rank1 = torch.sum(mask1 * rank1, dim=-1)
        rank2 = torch.sum(mask2 * rank2, dim=-1)

        if use_kernel:
            mask1 = torch.sum(mask1, dim=-1)
            mask2 = torch.sum(mask2, dim=-1)

            mask = torch.stack([mask1, mask2], dim=0).to(torch.int32)
            dest_idx = torch.stack([top1_idx * capacity + rank1, top2_idx * capacity + rank2], dim=0).to(torch.int32)

            return used_capacity, probs, mask, dest_idx, num_experts * capacity
        else:
            """
            The following code is equivalent to:

                ```
                weight1 = mask1 * probs.type_as(inputs)
                weight2 = mask2 * probs.type_as(inputs)
                rank1_sc = F.one_hot(rank1, num_classes=capacity)
                rank2_sc = F.one_hot(rank2, num_classes=capacity)

                cb_weight1 = weight1.unsqueeze(2) * rank1_sc.unsqueeze(1)
                cb_weight2 = weight2.unsqueeze(2) * rank2_sc.unsqueeze(1)
                cb_weight = cb_weight1 + cb_weight2
                sec_mask = cb_weight.bool()
                ```
            """

            weight1 = mask1 * probs.type_as(inputs)
            weight2 = mask2 * probs.type_as(inputs)

            cb_weight = torch.zeros(inputs.shape + (capacity,), device=inputs.device)
            sec_mask = torch.zeros_like(cb_weight, dtype=torch.bool)
            indices = torch.arange(0, inputs.shape[0], device=inputs.device)
            cb_weight[indices, top1_idx[indices], rank1[indices]] += weight1[indices, top1_idx[indices]]
            cb_weight[indices, top2_idx[indices], rank2[indices]] += weight2[indices, top2_idx[indices]]
            sec_mask[indices, top1_idx[indices], rank1[indices]] |= mask1.bool()[indices, top1_idx[indices]]
            sec_mask[indices, top2_idx[indices], rank2[indices]] |= mask2.bool()[indices, top2_idx[indices]]

            return used_capacity, cb_weight, sec_mask


class TopKRouter(MoeRouter):
    """Masked matmul router using tokens choose top-k experts assignment.

    NOTE: this is modified from flaxformer.
    This router uses the same mechanism as in Switch Transformer
    (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are
    sorted by router_probs and then routed to their choice of expert until the
    expert's expert_capacity is reached. There is no guarantee that each token is
    processed by an expert, or that each expert receives at least one token.

    Attributes:
        num_selected_experts: Maximum number of experts to which each token is
            routed. Tokens may be routed to fewer experts if particular experts are
            oversubscribed / reach capacity.
    """

    def __init__(
        self,
        num_selected_experts: int,
        capacity_factor_train: float = 1.25,
        capacity_factor_eval: float = 2.0,
        min_capacity: int = 4,
        noisy_func: Optional[Callable] = None,
        drop_tks: bool = True,
    ):
        super().__init__(
            num_selected_experts, capacity_factor_train, capacity_factor_eval, min_capacity, noisy_func, drop_tks
        )

    def forward(
        self,
        router_probs: torch.Tensor,
        expert_capacity: int,
    ) -> Tuple:
        """Computes masks for the top-k experts per token.

        Args:
            router_probs: <float32>[num_groups, tokens_per_group, num_experts]
                probabilities used to determine the routing of tokens to the experts.

        Returns:
            Dispatch and combine arrays for routing with masked matmuls.
        """
        # TODO: FIXME: add parallel group
        num_groups, _, num_experts = router_probs.shape

        # Top-k router probability and corresponding expert indices for each token.
        # Shape: [num_groups, tokens_per_group, num_selected_experts].
        expert_gate, expert_index = torch.topk(router_probs, self.k_value)

        self.set_aux_loss(router_probs, expert_index, num_experts)
        self.pop_router_loss()

        # Make num_selected_experts the leading axis to ensure that top-1 choices
        # have priority over top-2 choices, which have priority over top-3 choices,
        # etc.
        expert_index = torch.transpose(expert_index, 1, 2)
        # Shape: [num_groups, num_selected_experts * tokens_per_group]
        expert_index = expert_index.reshape(num_groups, -1)

        # Create mask out of indices.
        # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
        expert_mask = F.one_hot(expert_index, num_experts).to(torch.int32)

        # Experts have a fixed capacity that we cannot exceed. A token's priority
        # within the expert's buffer is given by the masked, cumulative capacity of
        # its target expert.
        # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
        token_priority = torch.cumsum(expert_mask, dim=1) * expert_mask - 1
        # Shape: [num_groups, num_selected_experts, tokens_per_group, num_experts].
        token_priority = token_priority.reshape((num_groups, self.k_value, -1, num_experts))
        # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
        token_priority = torch.transpose(token_priority, 1, 2)
        # For each token, across all selected experts, select the only non-negative
        # (unmasked) priority. Now, for group G routing to expert E, token T has
        # non-negative priority (i.e. token_priority[G,T,E] >= 0) if and only if E
        # is its targeted expert.
        # Shape: [num_groups, tokens_per_group, num_experts].
        token_priority = torch.max(token_priority, dim=2)[0]

        # Token T can only be routed to expert E if its priority is positive and
        # less than the expert capacity. One-hot matrix will ignore indices outside
        # the range [0, expert_capacity).
        # Shape: [num_groups, tokens_per_group, num_experts, expert_capacity].
        valid_mask = torch.logical_and(token_priority >= 0, token_priority < expert_capacity)
        token_priority = torch.masked_fill(token_priority, ~valid_mask, 0)
        dispatch_mask = F.one_hot(token_priority, expert_capacity).to(torch.bool)
        valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, -1, expert_capacity)
        dispatch_mask = torch.masked_fill(dispatch_mask, ~valid_mask, 0)

        # The combine array will be used for combining expert outputs, scaled by the
        # router probabilities. Shape: [num_groups, tokens_per_group, num_experts,
        # expert_capacity].
        combine_array = torch.einsum("...te,...tec->...tec", router_probs, dispatch_mask)

        return combine_array, dispatch_mask


def get_router_cls(top_k: int, grouped: bool = False) -> MoeRouter:
    if not grouped:
        if top_k == 1:
            return Top1Router
        elif top_k == 2:
            return Top2Router
        else:
            raise NotImplementedError("top_k > 2 is not supported yet")
    else:
        return TopKRouter
