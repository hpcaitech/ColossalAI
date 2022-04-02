import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.utils import get_current_device
from ._operation import COL_MOE_KERNEL_FLAG, AllToAll, AllGather, ReduceScatter, MoeDispatch, MoeCombine, moe_cumsum
from .experts import MoeExperts, Experts
from .utils import ForceFP32Parameter, UniformNoiseGenerator, NormalNoiseGenerator, autocast_softmax
from colossalai.zero.init_ctx import no_shard_zero_context, no_shard_zero_decrator
from typing import Callable, Optional, Type
from torch.distributed import ProcessGroup


class Top1Router(nn.Module):
    """Top1 router that returns the dispatch mask [s, e, c] and combine weight [s, e, c]
    for routing usage. More deailted function can be found in the paper about Switch Transformer
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

    def forward(self, inputs: torch.Tensor, use_kernel: bool = False, ep_group: Optional[ProcessGroup] = None):

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
            MOE_CONTEXT.add_loss(l_aux)
        elif not self.drop_tks:
            max_num = torch.max(torch.sum(mask, dim=0))
            dist.all_reduce(max_num, op=dist.ReduceOp.MAX, group=ep_group)
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


class Top2Router(nn.Module):
    """Top2 router that returns the dispatch mask [s, e, c] and combine weight [s, e, c]
    for routing usage. More deailted function can be found in the paper about ViT-MoE.

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

    def forward(self, inputs: torch.Tensor, use_kernel: bool = False, ep_group: Optional[ProcessGroup] = None):
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
            MOE_CONTEXT.add_loss(l_aux)
        elif not self.drop_tks:
            max_num = torch.max(torch.sum(cmask, dim=0))
            dist.all_reduce(max_num, op=dist.ReduceOp.MAX, group=ep_group)
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


class FP32LinearGate(nn.Module):
    """Gate module used in MOE layer. Just a linear function without bias.
    But it should be kept as fp32 forever.

    Args:
        d_model (int): Hidden dimension of training model
        num_experts (int): The number experts

    Attributes:
        weight (ForceFP32Parameter): The weight of linear gate
    """

    def __init__(self, d_model: int, num_experts: int, scale: float = 0.1):
        super().__init__()
        self.weight = ForceFP32Parameter(torch.empty(num_experts, d_model, device=get_current_device()))
        nn.init.trunc_normal_(self.weight, std=math.sqrt(scale / d_model))

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight)


class MoeLayer(nn.Module):
    """A MoE layer, that puts its input tensor to its gate and uses the output logits
    to router all tokens, is mainly used to exchange all tokens for every expert across
    the moe tensor group by all to all comunication. Then it will get the output of all
    experts and exchange the output. At last returns the output of the moe system.

    Args:
        dim_model (int): Dimension of model.
        num_experts (int): The number of experts.
        router (:class:`torch.nn.Module`): Instance of router used in routing.
        experts (:class:`torch.nn.Module`): Instance of experts generated by Expert.
    """

    @no_shard_zero_decrator(is_replicated=True)
    def __init__(self, dim_model: int, num_experts: int, router: nn.Module, experts: MoeExperts):
        super().__init__()
        self.d_model = dim_model
        self.num_experts = num_experts
        self.gate = FP32LinearGate(dim_model, num_experts)
        self.router = router
        self.experts = experts
        self.use_kernel = True if COL_MOE_KERNEL_FLAG and MOE_CONTEXT.use_kernel_optim else False
        self.ep_group = experts.dist_info.ep_group
        self.ep_size = experts.dist_info.ep_size
        self.num_local_experts = experts.num_local_experts

    def a2a_process(self, dispatch_data: torch.Tensor):
        expert_input = AllToAll.apply(dispatch_data, self.ep_group)

        input_shape = expert_input.shape

        expert_input = expert_input.reshape(self.ep_size, self.num_local_experts, -1, self.d_model)

        expert_output = self.experts(expert_input)
        expert_output = expert_output.reshape(input_shape)

        expert_output = AllToAll.apply(expert_output, self.ep_group)
        return expert_output

    def tp_process(self, dispatch_data: torch.Tensor):
        expert_in = AllGather.apply(dispatch_data, self.ep_group)
        expert_out = self.experts(expert_in)
        expert_out = ReduceScatter.apply(expert_out, self.ep_group)
        return expert_out

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = inputs.reshape(-1, self.d_model)
        fp32_input = tokens.to(torch.float32) if inputs.dtype != torch.float32 else tokens
        gate_output = self.gate(fp32_input)
        router_res = self.router(inputs=gate_output, use_kernel=self.use_kernel, ep_group=self.ep_group)

        if self.use_kernel:
            dispatch_data = MoeDispatch.apply(tokens, *router_res[1:])
            dispatch_data = dispatch_data.reshape(self.num_experts, -1, self.d_model)
        else:
            sec_mask_f = router_res[1].type_as(inputs)
            dispatch_data = torch.matmul(sec_mask_f.permute(1, 2, 0), tokens)

        # dispatch_data [e, c, h]
        if self.experts.comm_name == "all_to_all":
            expert_output = self.a2a_process(dispatch_data)
        elif self.experts.comm_name == "all_gather":
            expert_output = self.tp_process(dispatch_data)
        else:
            raise NotImplementedError("This kind of communication has not been implemented yet.\n Please use Experts "
                                      "build function.")
        # expert_output [e, c, h]

        if self.use_kernel:
            expert_output = expert_output.reshape(-1, self.d_model)
            ans = MoeCombine.apply(expert_output, *router_res)
        else:
            combine_weights = router_res[0].type_as(inputs)
            combine_weights = combine_weights.view(combine_weights.shape[0], -1)
            expert_output = expert_output.view(-1, expert_output.shape[-1])
            ans = torch.matmul(combine_weights, expert_output)

        ans = ans.reshape(inputs.shape)
        return ans


class MoeModule(nn.Module):
    """A class for users to create MoE modules in their models.

    Args:
        dim_model (int): Hidden dimension of training model
        num_experts (int): The number experts
        top_k (int, optional): The number of experts for dispatchment of each token
        capacity_factor_train (float, optional): Capacity factor in routing during training
        capacity_factor_eval (float, optional): Capacity factor in routing during evaluation
        min_capacity (int, optional): The minimum number of the capacity of each expert
        noisy_policy (str, optional): The policy of noisy function. Now we have 'Jitter' and 'Gaussian'.
            'Jitter' can be found in `Switch Transformer paper`_.
            'Gaussian' can be found in `ViT-MoE paper`_.
        drop_tks (bool, optional): Whether drops tokens in evaluation
        use_residual (bool, optional): Makes this MoE layer a Residual MoE.
            More information can be found in `Microsoft paper`_.
        residual_instance (nn.Module, optional): The instance of residual module in Resiual MoE
        expert_instance (MoeExperts, optional): The instance of experts module in MoeLayer
        expert_cls (Type[nn.Module], optional): The class of each expert when no instance is given
        expert_args (optional): The args of expert when no instance is given

    .. _Switch Transformer paper:
        https://arxiv.org/abs/2101.03961
    .. _ViT-MoE paper:
        https://arxiv.org/abs/2106.05974
    .. _Microsoft paper:
        https://arxiv.org/abs/2201.05596
    """

    def __init__(self,
                 dim_model: int,
                 num_experts: int,
                 top_k: int = 1,
                 capacity_factor_train: float = 1.25,
                 capacity_factor_eval: float = 2.0,
                 min_capacity: int = 4,
                 noisy_policy: Optional[str] = None,
                 drop_tks: bool = True,
                 use_residual: bool = False,
                 residual_instance: Optional[nn.Module] = None,
                 expert_instance: Optional[MoeExperts] = None,
                 expert_cls: Optional[Type[nn.Module]] = None,
                 **expert_args):
        super().__init__()

        noisy_func = None
        if noisy_policy is not None:
            if noisy_policy == 'Jitter':
                noisy_func = UniformNoiseGenerator()
            elif noisy_policy == 'Gaussian':
                noisy_func = NormalNoiseGenerator(num_experts)
            else:
                raise NotImplementedError("Unsupported input noisy policy")

        if top_k == 1:
            moe_router_cls = Top1Router
        elif top_k == 2:
            moe_router_cls = Top2Router
        else:
            raise NotImplementedError("top_k > 2 is not supported yet")

        self.moe_router = moe_router_cls(capacity_factor_train=capacity_factor_train,
                                         capacity_factor_eval=capacity_factor_eval,
                                         min_capacity=min_capacity,
                                         noisy_func=noisy_func,
                                         drop_tks=drop_tks)
        self.use_residual = use_residual
        if use_residual:
            if residual_instance is not None:
                self.residual_module = residual_instance
            else:
                assert expert_cls is not None, \
                    "Expert class can't be None when residual instance is not given"
                self.residual_module = expert_cls(**expert_args)

            with no_shard_zero_context():
                self.residual_combine = nn.Linear(dim_model, 2, device=get_current_device())

        if expert_instance is not None:
            self.experts = expert_instance
        else:
            assert expert_cls is not None, \
                "Expert class can't be None when experts instance is not given"
            self.experts = Experts(expert_cls, num_experts, **expert_args)

        self.moe_layer = MoeLayer(dim_model=dim_model,
                                  num_experts=num_experts,
                                  router=self.moe_router,
                                  experts=self.experts)

    def forward(self, inputs: torch.Tensor):
        moe_output = self.moe_layer(inputs)

        if self.use_residual:
            residual_output = self.residual_module(inputs)
            combine_coef = self.residual_combine(inputs)
            combine_coef = F.softmax(combine_coef, dim=-1)
            output = moe_output * combine_coef[..., 0:1] + residual_output * combine_coef[..., 1:]
        else:
            output = moe_output

        return output
