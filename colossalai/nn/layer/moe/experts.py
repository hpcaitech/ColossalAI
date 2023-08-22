import math
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.context import ParallelMode, seed
from colossalai.context.moe_context import MOE_CONTEXT
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
        expert_parallel: str,
        activation: str = None,
        drop_rate: float = 0,
    ):
        super().__init__()
        assert expert_parallel in ["EP", "TP"]
        self.expert_parallel = expert_parallel

        # get local and total experts
        self.num_total_experts = num_experts
        self.num_local_experts, self.moe_info = MOE_CONTEXT.get_info(num_experts,
                                                                     use_tp=True if expert_parallel == "TP" else False)

        # get settings for different parallel
        if expert_parallel == "TP":
            assert intermediate_size % MOE_CONTEXT.max_ep_size == 0, \
                "intermediate_size should be divide by maximum expert parallel size"
            intermediate_size = intermediate_size // MOE_CONTEXT.max_ep_size
            num_experts = self.num_total_experts
        else:
            num_experts = self.num_local_experts

        self.wi = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.wo = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))

        with seed(ParallelMode.TENSOR):
            nn.init.trunc_normal_(self.wi, std=math.sqrt(0.1 / hidden_size))
            nn.init.trunc_normal_(self.wo, std=math.sqrt(0.1 / intermediate_size))

        self.act = nn.GELU() if activation is None else activation
        self.drop = nn.Dropout(p=drop_rate)

        for param in self.parameters():
            set_moe_tensor_info(param, self.moe_info)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # inputs [g, e, c, h]

        e = x.size(1)
        h = x.size(-1)

        x = x.transpose(0, 1)
        inshape = x.shape
        x = x.reshape(e, -1, h)

        x = torch.bmm(x, self.wi)
        x = self.act(x)
        with seed(ParallelMode.TENSOR):
            x = self.drop(x)
        x = torch.bmm(x, self.wo)

        x = x.reshape(inshape)
        x = x.transpose(0, 1).contiguous()
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
                 drop_rate: float = 0):
        super().__init__(num_experts, hidden_size, intermediate_size, "EP", activation, drop_rate)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        dp_rank = dist.get_rank(get_dp_group(self))
        ep_rank = dist.get_rank(get_ep_group(self))
        ep_size = get_ep_size(self)
        # dp rank 0 will save the state dict
        if dp_rank == 0:
            for name, module in self.named_parameters():
                if module is self:
                    continue
                # create buffer
                buffer_module = deepcopy(module)
                # gather param from every ep rank
                for source_rank in range(ep_size):
                    current_prefix = f"{prefix}{source_rank}."
                    if ep_rank == source_rank:
                        dist.broadcast(module.data, src=source_rank, group=self.moe_info.ep_group)
                    else:
                        dist.broadcast(buffer_module.data, src=source_rank, group=self.moe_info.ep_group)
                    if ep_rank == 0:
                        destination[current_prefix + name] = buffer_module.data.cpu()

        dist.barrier()


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
                 drop_rate: float = 0):
        super().__init__(num_experts, hidden_size, intermediate_size, "TP", activation, drop_rate)


def get_expert_class(name: str) -> BaseMLPExperts:
    if name == "TP":
        return TPMLPExperts
    elif name == "EP":
        return EPMLPExperts
    else:
        raise ValueError(f"Unknown expert class name: {name}")
