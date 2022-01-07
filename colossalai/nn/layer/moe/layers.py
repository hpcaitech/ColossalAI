import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from colossalai.global_variables import moe_env
from colossalai.context import ParallelMode, seed
from colossalai.utils import get_current_device
from ._operation import AllToAll


class NormalNoiseGenerator:
    """Generates a random noisy mask for logtis tensor.
    All noise is generated from a normal distribution (0, 1 / E^2), where
    E = the number of experts.
    """

    def __init__(self, num_experts: int):
        self.normal = torch.distributions.normal.Normal(
            loc=torch.tensor(0.0, device=get_current_device()),
            scale=torch.tensor(1.0 / num_experts ** 2, device=get_current_device())
        ).rsample

    def __call__(self, inputs: torch.Tensor):
        noisy = self.normal(inputs.shape)
        return inputs + noisy


class Experts(nn.Module):
    """A wrapper class to create experts. It will create E experts across the
    moe model parallel group, where E is the number of experts. Every expert
    is a instence of the class, 'expert' in initialization parameters.
    """

    def __init__(self, expert, num_experts, **expert_args):
        super().__init__()

        assert num_experts % moe_env.model_parallel_size == 0, \
            "The number of experts should be divied by moe model size"

        num_local_experts = num_experts // moe_env.model_parallel_size
        with seed(ParallelMode.MOE_MODEL):
            self.experts = nn.ModuleList([
                expert(**expert_args) for _ in range(num_local_experts)])
        self.num_local_experts = num_local_experts
        for exp in self.experts:
            for param in exp.parameters():
                param.__setattr__('moe_param', 1)

    def forward(self, inputs):
        expert_input = torch.chunk(inputs, self.num_local_experts, dim=0)
        expert_output = []

        for i in range(self.num_local_experts):
            expert_output.append(self.experts[i](expert_input[i]))

        output = torch.cat(expert_output, dim=0)
        return output


class Top1Router(nn.Module):
    """Top1 router that returns the dispatch mask [s, e, c] and combine weight [s, e, c]
    for routing usage. More deailted function can be found in the paper about Switch Transformer
    of Google.
    """

    def __init__(self,
                 capacity_factor: float,
                 min_capacity: int,
                 noisy_func=None):
        super().__init__()
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.noisy_func = noisy_func
        self.uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(0.0, device=get_current_device()),
            high=torch.tensor(1.0, device=get_current_device())).rsample

    def get_capacity(self, logits_shape):
        capacity = math.ceil(self.capacity_factor *
                             logits_shape[0] / logits_shape[1])
        if capacity < self.min_capacity:
            capacity = self.min_capacity
        return capacity

    def forward(self, inputs):

        if self.noisy_func is not None:
            inputs_noisy = self.noisy_func(inputs)
        else:
            inputs_noisy = inputs

        logits = F.softmax(inputs, dim=1)

        num_experts = logits.shape[1]
        capacity = self.get_capacity(logits.shape)

        expert_idx = torch.argmax(inputs_noisy, dim=1)
        expert_mask = F.one_hot(expert_idx, num_classes=num_experts)
        expert_mask_f = expert_mask.float()

        exp_counts = torch.sum(expert_mask, dim=0).detach().to('cpu')

        me = torch.mean(logits, dim=0)
        ce = torch.mean(expert_mask_f, dim=0)
        l_aux = torch.sum(me * ce) * num_experts
        moe_env.add_loss(l_aux)

        rand_mask = expert_mask * self.uniform(logits.shape)
        _, dispatch_idx = torch.topk(rand_mask, k=capacity, dim=0)

        dispatch_mask = \
            expert_mask * torch.zeros_like(expert_mask).scatter_(0, dispatch_idx, 1)

        locations = torch.cumsum(dispatch_mask, dim=0) - 1
        locations = torch.sum(dispatch_mask * locations, dim=1)
        locations = F.one_hot(locations, num_classes=capacity)

        logits = logits * dispatch_mask
        combine_weights = logits.unsqueeze(2) * locations.unsqueeze(1)

        sec_mask = combine_weights.bool()
        return combine_weights, sec_mask, exp_counts


class Top2Router(nn.Module):
    """Top2 router that returns the dispatch mask [s, e, c] and combine weight [s, e, c]
    for routing usage. More deailted function can be found in the paper about ViT-MoE.
    """

    def __init__(self, capacity_factor: float, noisy_func=None):
        super().__init__()
        self.capacity_factor = capacity_factor
        self.noisy_func = noisy_func

    def get_capacity(self, logits_shape):
        capacity = math.ceil(2 * self.capacity_factor *
                             logits_shape[0] / logits_shape[1])
        return capacity

    def forward(self, inputs):
        if self.noisy_func is not None:
            inputs = self.noisy_func(inputs)

        logits = F.softmax(inputs, dim=-1)
        num_experts = logits.size(-1)
        capacity = self.get_capacity(logits.shape)

        _, expert_idx = torch.topk(logits, k=2, dim=-1, largest=True, sorted=True)
        top1_idx = expert_idx[:, 0]
        top2_idx = expert_idx[:, 1]

        mask1 = F.one_hot(top1_idx, num_classes=num_experts)
        mask2 = F.one_hot(top2_idx, num_classes=num_experts)

        loss_mask = (mask1 + mask2)
        exp_counts = torch.sum(loss_mask, dim=0).detach().to('cpu')
        me = torch.mean(logits, dim=0)
        ce = torch.mean(loss_mask.float(), dim=0)
        l_aux = num_experts * torch.sum(me * ce) / 2.0
        moe_env.add_loss(l_aux)

        locations1 = torch.cumsum(mask1, dim=0) - 1
        locations2 = torch.cumsum(mask2, dim=0) - 1
        locations2 += torch.sum(mask1, dim=0, keepdim=True)

        mask1 *= torch.lt(locations1, capacity)
        mask2 *= torch.lt(locations2, capacity)

        weight1 = mask1 * logits
        weight2 = mask2 * logits

        locations1 = torch.sum(mask1 * locations1, dim=1)
        locations2 = torch.sum(mask2 * locations2, dim=1)
        locations1_sc = F.one_hot(locations1, num_classes=capacity)
        locations2_sc = F.one_hot(locations2, num_classes=capacity)

        combine_weights1 = weight1.unsqueeze(2) * locations1_sc.unsqueeze(1)
        combine_weights2 = weight2.unsqueeze(2) * locations2_sc.unsqueeze(1)
        combine_weights = combine_weights1 + combine_weights2
        sec_mask = combine_weights.bool()

        return combine_weights, sec_mask, exp_counts


class MoeLayer(nn.Module):
    """A MoE layer, that puts its input tensor to its gate and uses the output logits
    to router all tokens, is mainly used to exchange all tokens for every expert across
    the moe tensor group by all to all comunication. Then it will get the output of all
    experts and exchange the output. At last returns the output of the moe system.
    """

    def __init__(self,
                 dim_model: int,
                 num_experts: int,
                 router: nn.Module,
                 experts: nn.Module):
        super().__init__()
        self.d_model = dim_model
        self.num_experts = num_experts
        self.gate = nn.Linear(dim_model, num_experts, device=get_current_device())
        self.router = router
        self.experts = experts

    def _router_part(self, tokens: torch.Tensor):
        gate_output = self.gate(tokens)
        return self.router(gate_output)

    def router_part(self, tokens: torch.Tensor):
        autocast_context = torch.is_autocast_enabled()
        if not autocast_context:
            return self._router_part(tokens)
        else:
            with autocast(enabled=False):
                if tokens.dtype == torch.float16:
                    input_tokens = tokens.float()
                else:
                    input_tokens = tokens
                return self._router_part(input_tokens)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = inputs.reshape(-1, self.d_model)

        combine_weights, sec_mask, exp_counts = self.router_part(tokens)

        sec_mask_f = sec_mask.type_as(inputs)
        dispatch_data = torch.matmul(sec_mask_f.permute(1, 2, 0), tokens)

        dispatch_data = AllToAll.apply(dispatch_data, ParallelMode.MOE_MODEL)

        expert_output = self.experts(dispatch_data)

        expert_output = AllToAll.apply(expert_output, ParallelMode.MOE_MODEL)

        combine_weights = combine_weights.view(combine_weights.shape[0], -1)
        expert_output = expert_output.view(-1, expert_output.shape[-1])

        ret = torch.matmul(combine_weights, expert_output)
        ret = ret.reshape(inputs.shape)

        return ret
