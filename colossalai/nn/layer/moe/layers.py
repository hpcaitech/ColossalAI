import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from colossalai.core import global_context as gpc
from colossalai.global_variables import moe_env
from colossalai.context import ParallelMode
from colossalai.utils import get_current_device
from ._operation import U_CUDA_MODE, AllToAll, MoeDispatch, MoeCombine, moe_cumsum
from .utils import autocast_softmax


class Top1Router(nn.Module):
    """Top1 router that returns the dispatch mask [s, e, c] and combine weight [s, e, c]
    for routing usage. More deailted function can be found in the paper about Switch Transformer
    of Google.

    :param capacity_factor: Capacity factor in routing
    :param min_capacity: The minimum number of the capacity of each expert
    :param noisy_func: Noisy function used in logits

    :type capacity_factor: float
    :type min_capacity: int
    :type noisy_func: Callable, optional
    """

    def __init__(self, capacity_factor: float, min_capacity: int = 0, select_policy: str = "first", noisy_func=None):
        super().__init__()
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.select_policy = select_policy
        self.noisy_func = noisy_func

        assert select_policy in {"first", "random"}
        if select_policy == "random":
            self.uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=get_current_device()),
                                                               high=torch.tensor(1.0,
                                                                                 device=get_current_device())).rsample

    def get_capacity(
        self,
        logits_shape,
    ):
        capacity = math.floor(self.capacity_factor * logits_shape[-2] / logits_shape[-1])
        capacity += capacity % 2
        capacity = max(capacity, self.min_capacity)
        assert capacity > 0
        return capacity

    def forward(self, inputs: torch.Tensor, cuda_mode: bool = False):

        if self.noisy_func is not None:
            inputs_noisy = self.noisy_func(inputs)
        else:
            inputs_noisy = inputs

        logits = autocast_softmax(inputs, dim=-1)
        num_experts = logits.size(-1)
        capacity = self.get_capacity(logits.shape)

        top1_idx = torch.argmax(inputs_noisy, dim=-1)
        mask = F.one_hot(top1_idx, num_classes=num_experts).to(torch.int32)

        if self.training:
            me = torch.mean(logits, dim=0)
            ce = torch.mean(mask.float(), dim=0)
            l_aux = num_experts * torch.sum(me * ce)
            moe_env.add_loss(l_aux)
        else:
            max_num = torch.max(torch.sum(mask, dim=0))
            dist.all_reduce(max_num, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.MOE_MODEL))
            capacity = max_num.item()

        if not self.training:
            ranks = moe_cumsum(mask)
        elif self.select_policy == "random":
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

        if cuda_mode:
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


class Top2Router(nn.Module):
    """Top2 router that returns the dispatch mask [s, e, c] and combine weight [s, e, c]
    for routing usage. More deailted function can be found in the paper about ViT-MoE.

    :param capacity_factor: Capacity factor in routing
    :param noisy_func: Noisy function used in logits

    :type capacity_factor: float
    :type noisy_func: Callable, optional
    """

    def __init__(self, capacity_factor: float, noisy_func=None):
        super().__init__()
        self.capacity_factor = capacity_factor
        self.noisy_func = noisy_func

    def get_capacity(self, logits_shape):
        capacity = math.floor(2 * self.capacity_factor * logits_shape[-2] / logits_shape[-1])
        capacity += capacity % 2
        assert capacity > 0
        return capacity

    def forward(self, inputs: torch.Tensor, cuda_mode: bool = False):
        # inputs: [s, h]
        if self.noisy_func is not None:
            inputs = self.noisy_func(inputs)

        logits = autocast_softmax(inputs, dim=-1)    # logits: [s, e]
        num_experts = logits.size(-1)
        capacity = self.get_capacity(logits.shape)

        top1_idx = torch.argmax(logits, dim=-1)
        mask1 = F.one_hot(top1_idx, num_classes=num_experts).to(torch.int32)
        logits_except1 = logits.masked_fill(mask1.bool(), float("-inf"))
        top2_idx = torch.argmax(logits_except1, dim=-1)
        mask2 = F.one_hot(top2_idx, num_classes=num_experts).to(torch.int32)

        cmask = (mask1 + mask2)    # loss: [s, e]
        if self.training:
            me = torch.mean(logits, dim=0)
            ce = torch.mean(cmask.float(), dim=0)
            l_aux = num_experts * torch.sum(me * ce) / 2.0
            moe_env.add_loss(l_aux)
        else:
            max_num = torch.max(torch.sum(cmask, dim=0))
            dist.all_reduce(max_num, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.MOE_MODEL))
            capacity = max_num.item()

        rank1 = moe_cumsum(mask1)    # rank1: [s, e]
        rank2 = moe_cumsum(mask2)
        rank2 += torch.sum(mask1, dim=-2, keepdim=True)

        mask1 *= torch.lt(rank1, capacity)
        mask2 *= torch.lt(rank2, capacity)

        rank1 = torch.sum(mask1 * rank1, dim=-1)
        rank2 = torch.sum(mask2 * rank2, dim=-1)

        if cuda_mode:
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


class MoeLayer(nn.Module):
    """A MoE layer, that puts its input tensor to its gate and uses the output logits
    to router all tokens, is mainly used to exchange all tokens for every expert across
    the moe tensor group by all to all comunication. Then it will get the output of all
    experts and exchange the output. At last returns the output of the moe system.

    :param dim_model: Dimension of model
    :param num_experts: The number of experts
    :param router: Instance of router used in routing
    :param experts: Instance of experts generated by Expert

    :type dim_model: int
    :type num_experts: int
    :type router: nn.Module
    :type experts: nn.Module
    """

    def __init__(self, dim_model: int, num_experts: int, router: nn.Module, experts: nn.Module):
        super().__init__()
        self.d_model = dim_model
        self.num_experts = num_experts
        self.gate = nn.Linear(dim_model, num_experts, bias=False, device=get_current_device())
        self.router = router
        self.experts = experts
        self.cuda_mode = True if U_CUDA_MODE and moe_env.enable_cuda else False

    def expert_part(self, expert_input: torch.Tensor):
        expert_input = AllToAll.apply(expert_input, ParallelMode.MOE_MODEL)

        input_shape = expert_input.shape

        expert_input = expert_input.reshape(moe_env.model_parallel_size,
                                            self.num_experts // moe_env.model_parallel_size, -1, self.d_model)

        expert_output = self.experts(expert_input)
        expert_output = expert_output.reshape(input_shape)

        expert_output = AllToAll.apply(expert_output, ParallelMode.MOE_MODEL)
        return expert_output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = inputs.reshape(-1, self.d_model)
        gate_output = self.gate(tokens)
        router_res = self.router(gate_output, self.cuda_mode)

        if self.cuda_mode:
            logits, mask, dest_idx, ec = router_res
            expert_input = MoeDispatch.apply(tokens, mask, dest_idx, ec)
            expert_output = self.expert_part(expert_input)
            ret = MoeCombine.apply(expert_output, logits, mask, dest_idx, ec)
        else:
            combine_weights, sec_mask = router_res
            sec_mask_f = sec_mask.type_as(inputs)
            expert_input = torch.matmul(sec_mask_f.permute(1, 2, 0), tokens)
            expert_output = self.expert_part(expert_input)
            combine_weights = combine_weights.view(combine_weights.shape[0], -1)
            expert_output = expert_output.view(-1, expert_output.shape[-1])
            ret = torch.matmul(combine_weights, expert_output)

        ret = ret.reshape(inputs.shape)
        return ret
