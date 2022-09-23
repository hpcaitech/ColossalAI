import math

import torch
import torch.nn as nn
from colossalai.context import ParallelMode, seed
from colossalai.utils import get_current_device
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.zero.init_ctx import no_shard_zero_decrator
from typing import Type


class MoeExperts(nn.Module):
    """Basic class for experts in MoE. It stores what kind of communication expersts use
    to exchange tokens, how many experts in a single GPU and parallel information such as
    expert parallel size, data parallel size and their distributed communication groups.
    """

    def __init__(self, comm_name: str, num_experts: int):
        super().__init__()
        assert comm_name in {"all_to_all", "all_gather"}, \
            "This kind of communication has not been implemented yet.\n Please use Experts build function."
        self.comm_name = comm_name
        # Get the configuration of experts' deployment and parallel information from moe contex
        self.num_local_experts, self.dist_info = MOE_CONTEXT.get_info(num_experts)


@no_shard_zero_decrator(is_replicated=False)
class Experts(MoeExperts):
    """A wrapper class to create experts. It will create E experts across the
    moe model parallel group, where E is the number of experts. Every expert
    is a instence of the class, 'expert' in initialization parameters.

    Args:
        expert_cls (:class:`torch.nn.Module`): The class of all experts
        num_experts (int): The number of experts
        expert_args: Args used to initialize experts, the args could be found in corresponding expert class
    """

    def __init__(self, expert_cls: Type[nn.Module], num_experts: int, **expert_args):
        super().__init__("all_to_all", num_experts)

        # Use seed to make every expert different from others
        with seed(ParallelMode.TENSOR):
            self.experts = nn.ModuleList([expert_cls(**expert_args) for _ in range(self.num_local_experts)])

        # Attach parallel information for all parameters in Experts
        for exp in self.experts:
            for param in exp.parameters():
                param.__setattr__('moe_info', self.dist_info)

    def forward(self, inputs: torch.Tensor):
        # Split inputs for each expert
        expert_input = torch.chunk(inputs, self.num_local_experts, dim=1)
        expert_output = []

        # Get outputs from each expert
        for i in range(self.num_local_experts):
            expert_output.append(self.experts[i](expert_input[i]))

        # Concatenate all outputs together
        output = torch.cat(expert_output, dim=1).contiguous()
        return output


class FFNExperts(MoeExperts):
    """Use torch.bmm to speed up for multiple experts.
    """

    def __init__(self, num_experts: int, d_model: int, d_ff: int, activation=None, drop_rate: float = 0):
        super().__init__("all_to_all", num_experts)

        self.w1 = nn.Parameter(torch.empty(self.num_local_experts, d_model, d_ff, device=get_current_device()))
        self.b1 = nn.Parameter(torch.empty(self.num_local_experts, 1, d_ff, device=get_current_device()))

        self.w2 = nn.Parameter(torch.empty(self.num_local_experts, d_ff, d_model, device=get_current_device()))
        self.b2 = nn.Parameter(torch.empty(self.num_local_experts, 1, d_model, device=get_current_device()))

        s1 = math.sqrt(0.1 / d_model)
        s2 = math.sqrt(0.1 / d_ff)

        with seed(ParallelMode.TENSOR):
            nn.init.trunc_normal_(self.w1, std=s1)
            nn.init.trunc_normal_(self.b1, std=s1)
            nn.init.trunc_normal_(self.w2, std=s2)
            nn.init.trunc_normal_(self.b2, std=s2)

        self.act = nn.GELU() if activation is None else activation
        self.drop = nn.Dropout(p=drop_rate)

        for param in self.parameters():
            param.__setattr__('moe_info', self.dist_info)

    def forward(self, inputs):    # inputs [g, el, c, h]

        el = inputs.size(1)
        h = inputs.size(-1)

        inputs = inputs.transpose(0, 1)
        inshape = inputs.shape
        inputs = inputs.reshape(el, -1, h)

        out_ff = torch.baddbmm(self.b1, inputs, self.w1)
        out_act = self.act(out_ff)
        with seed(ParallelMode.TENSOR):
            out_inter = self.drop(out_act)

        out_model = torch.baddbmm(self.b2, out_inter, self.w2)
        with seed(ParallelMode.TENSOR):
            outputs = self.drop(out_model)    # outputs [el, gc, h]

        outputs = outputs.reshape(inshape)
        outputs = outputs.transpose(0, 1).contiguous()
        return outputs


class TPExperts(MoeExperts):
    """Use tensor parallelism to split each expert evenly, which can deploy experts in
    case that the number of experts can't be divied by maximum expert parallel size or
    maximum expert parallel size can't be divied by the number of experts.
    """

    def __init__(self, num_experts: int, d_model: int, d_ff: int, activation=None, drop_rate: float = 0):
        super().__init__("all_gather", MOE_CONTEXT.max_ep_size)

        assert d_ff % MOE_CONTEXT.max_ep_size == 0, \
            "d_ff should be divied by maximum expert parallel size"

        p_ff = d_ff // MOE_CONTEXT.max_ep_size

        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, p_ff, device=get_current_device()))
        self.b1 = nn.Parameter(torch.empty(num_experts, 1, p_ff, device=get_current_device()))

        self.w2 = nn.Parameter(torch.empty(num_experts, p_ff, d_model, device=get_current_device()))
        self.b2 = nn.Parameter(torch.empty(num_experts, 1, d_model, device=get_current_device()))

        s1 = math.sqrt(0.1 / d_model)
        s2 = math.sqrt(0.1 / d_ff)

        with seed(ParallelMode.TENSOR):
            nn.init.trunc_normal_(self.w1, std=s1)
            nn.init.trunc_normal_(self.b1, std=s1)
            nn.init.trunc_normal_(self.w2, std=s2)

        nn.init.trunc_normal_(self.b2, std=s2)

        self.act = nn.GELU() if activation is None else activation
        self.drop = nn.Dropout(p=drop_rate)

        self.w1.__setattr__('moe_info', self.dist_info)
        self.w2.__setattr__('moe_info', self.dist_info)
        self.b1.__setattr__('moe_info', self.dist_info)

    def forward(self, inputs):    # inputs [g, e, c, h]

        e = inputs.size(1)
        h = inputs.size(-1)

        inputs = inputs.transpose(0, 1)
        inshape = inputs.shape
        inputs = inputs.reshape(e, -1, h)

        out_ff = torch.baddbmm(self.b1, inputs, self.w1)
        out_act = self.act(out_ff)
        with seed(ParallelMode.TENSOR):
            out_inter = self.drop(out_act)

        out_model = torch.baddbmm(self.b2, out_inter, self.w2)
        outputs = self.drop(out_model)    # outputs [e, gc, h]

        outputs = outputs.reshape(inshape)
        outputs = outputs.transpose(0, 1).contiguous()
        return outputs    # outputs [g, e, c, h]
