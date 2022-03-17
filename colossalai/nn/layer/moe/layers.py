import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from colossalai.core import global_context as gpc
from colossalai.global_variables import moe_env
from colossalai.context import ParallelMode
from colossalai.utils import get_current_device
from ._operation import U_CUDA_MODE, AllToAll, AllGather, ReduceScatter, MoeDispatch, MoeCombine, moe_cumsum
from .experts import MoeExperts
from .utils import autocast_softmax
from typing import Callable


class Top1Router(nn.Module):
    """Top1 router that returns the dispatch mask [s, e, c] and combine weight [s, e, c]
    for routing usage. More deailted function can be found in the paper about Switch Transformer
    of Google.

    :param capacity_factor_train: Capacity factor in routing of training
    :param capacity_factor_eval: Capacity factor in routing of evaluation
    :param min_capacity: The minimum number of the capacity of each expert
    :param select_policy: The policy about tokens selection
    :param noisy_func: Noisy function used in logits
    :param drop_tks: Whether drops tokens in evaluation

    :type capacity_factor_train: float, optional
    :type capacity_factor_eval: float, optional
    :type min_capacity: int, optional
    :type select_policy: str, optional
    :type noisy_func: Callable, optional
    :type drop_tks: bool, optional
    """

    def __init__(self,
                 capacity_factor_train: float = 1.25,
                 capacity_factor_eval: float = 2.0,
                 min_capacity: int = 4,
                 select_policy: str = "first",
                 noisy_func: Callable = None,
                 drop_tks: bool = True):
        super().__init__()
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.min_capacity = min_capacity
        self.select_policy = select_policy
        self.noisy_func = noisy_func
        self.drop_tks = drop_tks

        assert select_policy in {"first", "random"}
        if select_policy == "random":
            self.uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=get_current_device()),
                                                               high=torch.tensor(1.0,
                                                                                 device=get_current_device())).rsample

    def get_capacity(
        self,
        logits_shape,
    ):
        capacity_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        capacity = math.floor(capacity_factor * logits_shape[-2] / logits_shape[-1])
        capacity += capacity % 2
        capacity = max(capacity, self.min_capacity)
        assert capacity > 0
        return capacity

    def forward(self, inputs: torch.Tensor, cuda_mode: bool = False):

        if self.noisy_func is not None and self.training:
            inputs = self.noisy_func(inputs)

        logits = autocast_softmax(inputs, dim=-1)
        num_experts = logits.size(-1)
        capacity = self.get_capacity(logits.shape)

        top1_idx = torch.argmax(inputs, dim=-1)
        mask = F.one_hot(top1_idx, num_classes=num_experts).to(torch.int32)

        if self.training:
            me = torch.mean(logits, dim=0)
            ce = torch.mean(mask.float(), dim=0)
            l_aux = num_experts * torch.sum(me * ce)
            moe_env.add_loss(l_aux)
        elif not self.drop_tks:
            max_num = torch.max(torch.sum(mask, dim=0))
            dist.all_reduce(max_num, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.MOE_MODEL))
            capacity = max_num.item()
        else:
            pass

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

    :param capacity_factor_train: Capacity factor in routing of training
    :param capacity_factor_eval: Capacity factor in routing of evaluation
    :param min_capacity: The minimum number of the capacity of each expert
    :param noisy_func: Noisy function used in logits
    :param drop_tks: Whether drops tokens in evaluation

    :type capacity_factor_train: float, optional
    :type capacity_factor_eval: float, optional
    :type min_capacity: int, optional
    :type noisy_func: Callable, optional
    :type drop_tks: bool, optional
    """

    def __init__(self,
                 capacity_factor_train: float = 1.25,
                 capacity_factor_eval: float = 2.0,
                 min_capacity: int = 4,
                 noisy_func: Callable = None,
                 drop_tks: bool = True):
        super().__init__()
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.min_capacity = min_capacity
        self.noisy_func = noisy_func
        self.drop_tks = drop_tks

    def get_capacity(
        self,
        logits_shape,
    ):
        capacity_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        capacity = math.floor(capacity_factor * logits_shape[-2] / logits_shape[-1])
        capacity += capacity % 2
        capacity = max(capacity, self.min_capacity)
        assert capacity > 0
        return capacity

    def forward(self, inputs: torch.Tensor, cuda_mode: bool = False):
        # inputs: [s, h]
        if self.noisy_func is not None and self.training:
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
            l_aux = num_experts * torch.sum(me * ce) / 2.0    # div 2 to normalize it to 1
            moe_env.add_loss(l_aux)
        elif not self.drop_tks:
            max_num = torch.max(torch.sum(cmask, dim=0))
            dist.all_reduce(max_num, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.MOE_MODEL))
            capacity = max_num.item()
        else:
            pass

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

    def __init__(self, dim_model: int, num_experts: int, router: nn.Module, experts: MoeExperts):
        super().__init__()
        self.d_model = dim_model
        self.num_experts = num_experts
        self.gate = nn.Linear(dim_model, num_experts, bias=False, device=get_current_device())
        self.router = router
        self.experts = experts
        self.cuda_mode = True if U_CUDA_MODE and moe_env.enable_cuda else False

    def a2a_process(self, dispatch_data: torch.Tensor):
        expert_input = AllToAll.apply(dispatch_data, ParallelMode.MOE_MODEL)

        input_shape = expert_input.shape

        expert_input = expert_input.reshape(moe_env.model_parallel_size,
                                            self.num_experts // moe_env.model_parallel_size, -1, self.d_model)

        expert_output = self.experts(expert_input)
        expert_output = expert_output.reshape(input_shape)

        expert_output = AllToAll.apply(expert_output, ParallelMode.MOE_MODEL)
        return expert_output

    def tp_process(self, dispatch_data: torch.Tensor):
        expert_in = AllGather.apply(dispatch_data, ParallelMode.MOE_MODEL)
        expert_out = self.experts(expert_in)
        expert_out = ReduceScatter.apply(expert_out, ParallelMode.MOE_MODEL)
        return expert_out

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = inputs.reshape(-1, self.d_model)
        gate_output = self.gate(tokens)
        router_res = self.router(gate_output, self.cuda_mode)

        if self.cuda_mode:
            dispatch_data = MoeDispatch.apply(tokens, *router_res[1:])
            dispatch_data = dispatch_data.reshape(self.num_experts, -1, self.d_model)
        else:
            sec_mask_f = router_res[1].type_as(inputs)
            dispatch_data = torch.matmul(sec_mask_f.permute(1, 2, 0), tokens)

        # dispatch_data [e, c, h]
        if self.experts.comm == "all_to_all":
            expert_output = self.a2a_process(dispatch_data)
        elif self.experts.comm == "all_gather":
            expert_output = self.tp_process(dispatch_data)
        else:
            raise NotImplementedError("This kind of communication has not been implemented yet.\n Please use Experts "
                                      "build function.")
        # expert_output [e, c, h]

        if self.cuda_mode:
            expert_output = expert_output.reshape(-1, self.d_model)
            ans = MoeCombine.apply(expert_output, *router_res)
        else:
            combine_weights = router_res[0]
            combine_weights = combine_weights.view(combine_weights.shape[0], -1)
            expert_output = expert_output.view(-1, expert_output.shape[-1])
            ans = torch.matmul(combine_weights, expert_output)

        ans = ans.reshape(inputs.shape)
        return ans
