import math
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.context import ParallelMode, seed
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.nn.layer.moe._operation import MoeInGradScaler, MoeOutGradScaler
from colossalai.nn.layer.moe.utils import get_activation
from colossalai.tensor.moe_tensor.api import get_dp_group, get_ep_group, get_ep_size, set_moe_tensor_info


class BaseMLPExperts(nn.Module):
    """
    SparseMLP is a multi-layer perceptron with sparse expert parallel layers.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        expert_parallel: str = None,
        activation: str = None,
        drop_rate: float = 0,
        gated: bool = False,
    ):
        super().__init__()
        assert expert_parallel in ["EP", "TP", None]
        self.expert_parallel = expert_parallel
        self.num_total_experts = num_experts
        self.gated = gated

        # get expert parallel info
        if expert_parallel is not None:
            self.num_local_experts, self.moe_info = MOE_CONTEXT.get_info(
                num_experts, use_tp=True if expert_parallel == "TP" else False)
            # get settings for different parallel
            if expert_parallel == "TP":
                assert intermediate_size % MOE_CONTEXT.max_ep_size == 0, \
                    "intermediate_size should be divide by maximum expert parallel size"
                intermediate_size = intermediate_size // MOE_CONTEXT.max_ep_size
                num_experts = self.num_total_experts
            else:
                num_experts = self.num_local_experts
            self.ep_size = get_ep_size(self)
        else:
            self.num_local_experts = self.num_total_experts
            self.ep_size = 1

        if gated:
            self.wi_gate = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size * 2))
            self.wi_up = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        else:
            self.wi = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.wo = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))

        if expert_parallel is not None:
            with seed(ParallelMode.TENSOR):
                if gated:
                    nn.init.trunc_normal_(self.wi_gate, std=math.sqrt(0.1 / hidden_size))
                    nn.init.trunc_normal_(self.wi_up, std=math.sqrt(0.1 / hidden_size))
                else:
                    nn.init.trunc_normal_(self.wi, std=math.sqrt(0.1 / hidden_size))
                nn.init.trunc_normal_(self.wo, std=math.sqrt(0.1 / intermediate_size))

        self.act = get_activation(activation)
        self.drop = nn.Dropout(p=drop_rate)

        if expert_parallel is not None:
            for param in self.parameters():
                set_moe_tensor_info(param, self.moe_info)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # inputs [g, e, c, h]
        x = MoeInGradScaler.apply(x, self.ep_size)

        e = x.size(1)
        h = x.size(-1)

        x = x.transpose(0, 1)
        inshape = x.shape
        x = x.reshape(e, -1, h)

        if self.gated:
            x = self.act(torch.bmm(x, self.wi_gate)) * torch.bmm(x, self.wi_up)
        else:
            x = torch.bmm(x, self.wi)
            x = self.act(x)

        if self.expert_parallel is not None:
            with seed(ParallelMode.TENSOR):
                x = self.drop(x)
        x = torch.bmm(x, self.wo)

        x = x.reshape(inshape)
        x = x.transpose(0, 1).contiguous()
        x = MoeOutGradScaler.apply(x, self.ep_size)
        return x    # outputs [g, e, c, h]


class EPMLPExperts(BaseMLPExperts):
    """
    Use expert parallelism to split each expert evenly, which can deploy experts in
    """

    def __init__(self,
                 num_experts: int,
                 hidden_size: int,
                 intermediate_size: int,
                 activation=None,
                 drop_rate: float = 0,
                 gated: bool = False):
        super().__init__(num_experts, hidden_size, intermediate_size, "EP", activation, drop_rate, gated)


class TPMLPExperts(BaseMLPExperts):
    """Use tensor parallelism to split each expert evenly, which can deploy experts in
    case that the number of experts can't be divide by maximum expert parallel size or
    maximum expert parallel size can't be divide by the number of experts.
    """

    def __init__(self,
                 num_experts: int,
                 hidden_size: int,
                 intermediate_size: int,
                 activation: str = None,
                 drop_rate: float = 0,
                 gated: bool = False):
        super().__init__(num_experts, hidden_size, intermediate_size, "TP", activation, drop_rate, gated)


def get_expert_class(name: str) -> BaseMLPExperts:
    if name == "TP":
        return TPMLPExperts
    elif name == "EP":
        return EPMLPExperts
    elif name is None:
        return BaseMLPExperts
    else:
        raise ValueError(f"Unknown expert class name: {name}")


def build_ffn_experts(num_experts: int, d_model: int, d_ff: int, activation=None, drop_rate: float = 0):
    mep_size = MOE_CONTEXT.max_ep_size
    if num_experts % mep_size == 0 or mep_size % num_experts == 0:
        return EPMLPExperts(num_experts, d_model, d_ff, activation, drop_rate)
    elif d_ff % mep_size == 0:
        return TPMLPExperts(num_experts, d_model, d_ff, activation, drop_rate)
    else:
        raise NotImplementedError(f"Can not build {num_experts} experts in {mep_size} GPUS.")
