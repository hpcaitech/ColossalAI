import math

import torch
import torch.nn as nn
from colossalai.global_variables import moe_env
from colossalai.context import ParallelMode, seed
from colossalai.utils import get_current_device


class Experts(nn.Module):
    """A wrapper class to create experts. It will create E experts across the
    moe model parallel group, where E is the number of experts. Every expert
    is a instence of the class, 'expert' in initialization parameters.

    :param expert: The class of all experts
    :param num_experts: The number of experts
    :param expert_args: Args used to initialize experts

    :type num_experts: int
    """

    def __init__(self, expert, num_experts, **expert_args):
        super().__init__()

        assert num_experts % moe_env.model_parallel_size == 0, \
            "The number of experts should be divied by moe model size"

        num_local_experts = num_experts // moe_env.model_parallel_size
        with seed(ParallelMode.MOE_MODEL):
            self.experts = nn.ModuleList([expert(**expert_args) for _ in range(num_local_experts)])
        self.num_local_experts = num_local_experts
        for exp in self.experts:
            for param in exp.parameters():
                param.__setattr__('moe_param', True)

    def forward(self, inputs):
        expert_input = torch.chunk(inputs, self.num_local_experts, dim=1)
        expert_output = []

        for i in range(self.num_local_experts):
            expert_output.append(self.experts[i](expert_input[i]))

        output = torch.cat(expert_output, dim=1).contiguous()
        return output


class FFNExperts(nn.Module):

    def __init__(self, num_experts: int, d_model: int, d_ff: int, activation=None, drop_rate: float = 0):
        super().__init__()

        assert num_experts % moe_env.model_parallel_size == 0, \
            "The number of experts should be divied by moe model size"

        num_local_experts = num_experts // moe_env.model_parallel_size

        self.w1 = nn.Parameter(torch.empty(num_local_experts, d_model, d_ff, device=get_current_device()))
        self.b1 = nn.Parameter(torch.empty(num_local_experts, 1, d_ff, device=get_current_device()))

        self.w2 = nn.Parameter(torch.empty(num_local_experts, d_ff, d_model, device=get_current_device()))
        self.b2 = nn.Parameter(torch.empty(num_local_experts, 1, d_model, device=get_current_device()))

        s1 = math.sqrt(0.1 / d_model)
        s2 = math.sqrt(0.1 / d_ff)

        with seed(ParallelMode.MOE_MODEL):
            nn.init.trunc_normal_(self.w1, std=s1)
            nn.init.trunc_normal_(self.b1, std=s1)
            nn.init.trunc_normal_(self.w2, std=s2)
            nn.init.trunc_normal_(self.b2, std=s2)

        self.act = nn.GELU() if activation is None else activation
        self.drop = nn.Dropout(p=drop_rate)

        for param in self.parameters():
            param.__setattr__('moe_param', True)

    def forward(self, inputs):    # x [g, el, c, h]

        el = inputs.size(1)
        h = inputs.size(-1)

        inputs = inputs.transpose(0, 1)
        inshape = inputs.shape
        inputs = inputs.reshape(el, -1, h)

        out_ff = torch.baddbmm(self.b1, inputs, self.w1)
        out_act = self.act(out_ff)
        with seed(ParallelMode.TENSOR):
            inter = self.drop(out_act)

        out_model = torch.baddbmm(self.b2, inter, self.w2)
        with seed(ParallelMode.TENSOR):
            outputs = self.drop(out_model)    # outputs [el, gc, h]

        outputs = outputs.reshape(inshape)
        outputs = outputs.transpose(0, 1).contiguous()
        return outputs
