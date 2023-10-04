import math
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from colossalai.moe._operation import AllGather, AllToAll, MoeCombine, MoeDispatch, ReduceScatter
from colossalai.moe.experts import BaseMLPExperts, get_expert_class
from colossalai.moe.manager import MOE_MANAGER
from colossalai.moe.routers import MoeRouter, get_router_cls
from colossalai.moe.utils import get_noise_generator
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.tensor.moe_tensor.api import get_ep_group, get_ep_size


class SparseMLP(nn.Module):
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
        residual_instance (nn.Module, optional): The instance of residual module in Residual MoE
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

    def __init__(
        self,
        num_experts: int,
        top_k: int = 1,
        capacity_factor_train: float = 1.25,
        capacity_factor_eval: float = 2.0,
        min_capacity: int = 4,
        noisy_policy: Optional[str] = None,
        drop_tks: bool = True,
        expert_parallel: str = "EP",
        hidden_size: int = 2048,
        intermediate_size: int = 2048,
        activation: str = None,
        gated: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.use_kernel = MOE_MANAGER.use_kernel_optim
        self.expert_parallel = expert_parallel
        self.gated = gated
        assert expert_parallel in [
            "EP",
            "TP",
            None,
        ], f"Unsupported expert parallel type {expert_parallel}"

        # moe router
        noisy_func = get_noise_generator(noisy_policy, num_experts)
        router_cls = get_router_cls(top_k)
        self.router: MoeRouter = router_cls(
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_eval,
            min_capacity=min_capacity,
            noisy_func=noisy_func,
            drop_tks=drop_tks,
        )

        # moe experts
        expert_cls = get_expert_class(expert_parallel)
        self.experts: BaseMLPExperts = expert_cls(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            gated=gated,
        )
        if expert_parallel is not None:
            self.ep_group = get_ep_group(self.experts)
            self.ep_size = get_ep_size(self.experts)
        else:
            self.ep_group = None
        self.num_local_experts = self.experts.num_local_experts

        # gate
        self.gate_weight = torch.nn.Parameter(torch.empty(num_experts, self.hidden_size))

        # init param
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # expert param should be different
        if self.expert_parallel is not None:
            seed_ctx = Randomizer(MOE_MANAGER.seed).fork_rng(enable_cpu=True)
        else:
            seed_ctx = nullcontext()
        with seed_ctx:
            if self.gated:
                torch.nn.init.normal_(self.experts.wi_gate, std=math.sqrt(0.1 / self.hidden_size))
                torch.nn.init.normal_(self.experts.wi_up, std=math.sqrt(0.1 / self.hidden_size))
            else:
                torch.nn.init.normal_(self.experts.wi, std=math.sqrt(0.1 / self.hidden_size))
            torch.nn.init.normal_(self.experts.wo, std=math.sqrt(0.1 / self.intermediate_size))
        torch.nn.init.normal_(self.gate_weight, std=math.sqrt(0.1 / self.hidden_size))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs (torch.Tensor): The input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # reshape the input tokens
        tokens = inputs.reshape(-1, self.hidden_size)

        # the data type of the inputs in the gating should be fp32
        fp32_input = tokens.to(torch.float)
        fp32_weight = self.gate_weight.to(torch.float)
        gate_output = F.linear(fp32_input, fp32_weight)

        # the result from the router
        route_result_list = self.router(inputs=gate_output, use_kernel=self.use_kernel, ep_group=self.ep_group)

        # dispatch_data: (num_experts, capacity, hidden_size)
        if self.use_kernel:
            dispatch_data = MoeDispatch.apply(tokens, *route_result_list[1:])
            dispatch_data = dispatch_data.reshape(self.num_experts, -1, self.hidden_size)
        else:
            sec_mask_f = route_result_list[1].type_as(inputs)
            dispatch_data = torch.matmul(sec_mask_f.permute(1, 2, 0), tokens)

        # expert_output: (num_groups, num_experts, capacity, hidden_size)
        if self.expert_parallel == "EP":
            expert_output = self._ep_process(dispatch_data)
        elif self.expert_parallel == "TP":
            expert_output = self._tp_process(dispatch_data)
        elif self.expert_parallel is None:
            expert_output = self._local_process(dispatch_data)
        else:
            raise NotImplementedError("This kind of communication has not been implemented yet.\n"
                                      "Please use Experts build function.")

        if self.use_kernel:
            expert_output = expert_output.reshape(-1, self.hidden_size)
            ans = MoeCombine.apply(expert_output, *route_result_list)
        else:
            combine_weights = route_result_list[0].type_as(inputs)
            combine_weights = combine_weights.view(combine_weights.shape[0], -1)
            expert_output = expert_output.view(-1, expert_output.shape[-1])
            ans = torch.matmul(combine_weights, expert_output)

        ans = ans.reshape(inputs.shape)
        return ans

    def _local_process(self, expert_in: torch.Tensor) -> torch.Tensor:
        expert_in = expert_in.unsqueeze(0)
        expert_out = self.experts(expert_in)
        return expert_out

    def _ep_process(self, dispatch_data: torch.Tensor) -> torch.Tensor:
        """
        Expert Parallel

        Args:
            dispatch_data (torch.Tensor): (num_experts, capacity, hidden_size)

        Returns:
            torch.Tensor: (num_experts, capacity, hidden_size)
        """
        expert_input = AllToAll.apply(dispatch_data, self.ep_group)
        input_shape = expert_input.shape
        expert_input = expert_input.reshape(self.ep_size, self.num_local_experts, -1, self.hidden_size)
        expert_output = self.experts(expert_input)
        expert_output = expert_output.reshape(input_shape)
        expert_output = AllToAll.apply(expert_output, self.ep_group)
        return expert_output

    def _tp_process(self, dispatch_data: torch.Tensor) -> torch.Tensor:
        """
        TP with overlap.

        origin:
                   |    C    |
        |     A    |         |    R    |

        overlap:
              |    C1   ||    C2   ||    C3   ||    C4   |
        | A1 || A2 |     | R1 | A3 || R2 | A4 || R3 |     | R4 |

        C is computation, A is all gather, R is reduce scatter.

        Args:
            dispatch_data (torch.Tensor): (num_experts, capacity, hidden_size)

        Returns:
            torch.Tensor: (num_experts, capacity, hidden_size)
        """
        chunk_num = 4
        chunk_size = dispatch_data.shape[0] // chunk_num
        out = torch.empty_like(dispatch_data)
        in_data = None
        in_handle = None
        out_data = None
        out_handle = None

        # backward compatibility for async op
        torch.cuda.synchronize()

        def get_chunk_slice(idx: int, gap: int) -> Tuple[slice]:
            return (slice(idx * gap, (idx + 1) * gap),)

        for i in range(chunk_num):
            cur_chunk_slice = get_chunk_slice(i, chunk_size)

            # if first, all gather
            if i == 0:
                d = dispatch_data[cur_chunk_slice].contiguous()
                expert_in, _ = AllGather.apply(d, self.ep_group)
            else:
                expert_in = in_data

            # async communication while compute
            if i != 0:
                # reduce scatter last out
                out_data, out_handle = ReduceScatter.apply(out_data, self.ep_group, True)
            if i != chunk_num - 1:
                # all gather next in
                next_d = dispatch_data[get_chunk_slice(i + 1, chunk_size)].contiguous()
                in_data, in_handle = AllGather.apply(next_d, self.ep_group, True)

            # compute
            expert_out = self.experts(expert_in, cur_chunk_slice)

            # sync handle
            if i != 0:
                out_handle.wait()
                out[get_chunk_slice(i - 1, chunk_size)] = out_data
            if i != chunk_num - 1:
                in_handle.wait()
            out_data = expert_out

            # store out for last loop
            if i == chunk_num - 1:
                out_data, _ = ReduceScatter.apply(out_data, self.ep_group)
                out[cur_chunk_slice] = out_data

            # sync for async op
            torch.cuda.synchronize()
        return out
