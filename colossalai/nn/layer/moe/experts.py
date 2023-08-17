import math

import torch
import torch.nn as nn

from colossalai.context import ParallelMode, seed
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.tensor.moe_tensor.api import set_moe_tensor_info
from colossalai.utils import get_current_device


class BaseExperts(nn.Module):
    """Basic class for experts in MoE. It stores what kind of communication experts use
    to exchange tokens, how many experts in a single GPU and parallel information such as
    expert parallel size, data parallel size and their distributed communication groups.
    """

    def __init__(self, comm_name: str, num_experts: int) -> None:
        super().__init__()
        assert comm_name in {"all_to_all", "all_gather"}, \
            "This kind of communication has not been implemented yet.\n Please use Experts build function."
        self.comm_name = comm_name
        self.num_total_experts = num_experts
        # Get the configuration of experts' deployment and parallel information from moe context
        self.num_local_experts, self.dist_info = MOE_CONTEXT.get_info(num_experts)


class EPExperts(BaseExperts):
    """Use torch.bmm to speed up for multiple experts.
    """

    def __init__(self,
                 num_experts: int,
                 hidden_size: int,
                 intermediate_size: int,
                 activation=None,
                 drop_rate: float = 0):
        super().__init__("all_to_all", num_experts)

        self.w1 = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, intermediate_size, device=get_current_device()))
        self.b1 = nn.Parameter(torch.empty(self.num_local_experts, 1, intermediate_size, device=get_current_device()))

        self.w2 = nn.Parameter(
            torch.empty(self.num_local_experts, intermediate_size, hidden_size, device=get_current_device()))
        self.b2 = nn.Parameter(torch.empty(self.num_local_experts, 1, hidden_size, device=get_current_device()))

        s1 = math.sqrt(0.1 / hidden_size)
        s2 = math.sqrt(0.1 / intermediate_size)

        with seed(ParallelMode.TENSOR):
            nn.init.trunc_normal_(self.w1, std=s1)
            nn.init.trunc_normal_(self.b1, std=s1)
            nn.init.trunc_normal_(self.w2, std=s2)
            nn.init.trunc_normal_(self.b2, std=s2)

        self.act = nn.GELU() if activation is None else activation
        self.drop = nn.Dropout(p=drop_rate)

        for param in self.parameters():
            set_moe_tensor_info(param, self.dist_info)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:    # inputs [g, el, c, h]

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


class TPExperts(BaseExperts):
    """Use tensor parallelism to split each expert evenly, which can deploy experts in
    case that the number of experts can't be divide by maximum expert parallel size or
    maximum expert parallel size can't be divide by the number of experts.
    """

    def __init__(self,
                 num_experts: int,
                 hidden_size: int,
                 intermediate_size: int,
                 activation=None,
                 drop_rate: float = 0):
        super().__init__("all_gather", MOE_CONTEXT.max_ep_size)

        assert intermediate_size % MOE_CONTEXT.max_ep_size == 0, \
            "d_ff should be divide by maximum expert parallel size"

        p_ff = intermediate_size // MOE_CONTEXT.max_ep_size

        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_size, p_ff, device=get_current_device()))
        self.b1 = nn.Parameter(torch.empty(num_experts, 1, p_ff, device=get_current_device()))

        self.w2 = nn.Parameter(torch.empty(num_experts, p_ff, hidden_size, device=get_current_device()))
        self.b2 = nn.Parameter(torch.empty(num_experts, 1, hidden_size, device=get_current_device()))

        s1 = math.sqrt(0.1 / hidden_size)
        s2 = math.sqrt(0.1 / intermediate_size)

        with seed(ParallelMode.TENSOR):
            nn.init.trunc_normal_(self.w1, std=s1)
            nn.init.trunc_normal_(self.b1, std=s1)
            nn.init.trunc_normal_(self.w2, std=s2)

        nn.init.trunc_normal_(self.b2, std=s2)

        self.act = nn.GELU() if activation is None else activation
        self.drop = nn.Dropout(p=drop_rate)

        for param in [self.w1, self.b1, self.w2]:
            set_moe_tensor_info(param, self.dist_info)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:    # inputs [g, e, c, h]

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


def get_expert_class(name: str) -> BaseExperts:
    if name == "TP":
        return TPExperts
    elif name == "EP":
        return EPExperts
    else:
        raise ValueError(f"Unknown expert class name: {name}")
