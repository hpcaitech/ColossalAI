import dataclasses
import math
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from colossalai.legacy.moe.load_balance import LoadBalancer
from colossalai.legacy.moe.utils import create_ep_hierarchical_group, get_noise_generator
from colossalai.moe._operation import AllGather, AllToAll, HierarchicalAllToAll, MoeCombine, MoeDispatch, ReduceScatter
from colossalai.shardformer.layer.moe import MLPExperts
from colossalai.tensor.moe_tensor.api import get_dp_group, get_ep_group, get_ep_group_ranks, get_ep_size


class SparseMLP(nn.Module):
    """A class for users to create MoE modules in their models.

    Args:
        dim_model (int): Hidden dimension of training model
        num_experts (int): The number experts
        top_k (int, optional): The number of experts for dispatchment of each token
        parallel (str): parallel mode. Should be "EP", "TP" or None
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
        hidden_size: int,
        intermediate_size: int,
        router_top_k: int = 1,
        parallel: str = "EP",
        router_loss: bool = True,
        router_norm: bool = False,
        router_capacity_factor_train: float = 1.25,
        router_capacity_factor_eval: float = 2.0,
        router_min_capacity: int = 4,
        router_noisy_policy: Optional[str] = None,
        router_drop_tks: bool = True,
        mlp_activation: Optional[str] = None,
        mlp_gated: bool = False,
        enable_load_balance: bool = False,
        load_balance_tolerance: float = 0.1,
        load_balance_beam_width: int = 8,
        load_balance_group_swap_factor: float = 0.4,
        enable_kernel: bool = False,
        enable_comm_overlap: bool = False,
        enable_hierarchical_comm: bool = True,
        return_gate_logits: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.gated = mlp_gated
        self.return_gate_logits = return_gate_logits
        self.enable_kernel = enable_kernel
        self.enable_comm_overlap = enable_comm_overlap
        # self.expert_parallel = MOE_MANAGER.get_parallel()
        assert parallel in ["EP", "TP", None], "parallel mode must be EP, TP or None"
        self.parallel = parallel
        self.router_loss = router_loss
        self.router_norm = router_norm

        # moe router
        noisy_func = get_noise_generator(router_noisy_policy, num_experts)
        router_cls = get_router_cls(router_top_k)
        self.topk = router_top_k
        self.router: MoeRouter = router_cls(
            capacity_factor_train=router_capacity_factor_train,
            capacity_factor_eval=router_capacity_factor_eval,
            min_capacity=router_min_capacity,
            noisy_func=noisy_func,
            drop_tks=router_drop_tks,
        )

        # gate
        self.gate_weight = torch.nn.Parameter(torch.empty(num_experts, self.hidden_size))

        # moe experts
        self.experts = MLPExperts(
            num_experts=self.num_experts,
            expert_parallel=self.parallel,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation=mlp_activation,
            gated=mlp_gated,
            use_kernel=self.enable_kernel,
        )

        # get parallel settings
        if self.parallel is not None:
            self.ep_group = get_ep_group(self.experts)
            self.ep_size = get_ep_size(self.experts)
            self.ep_hierarchical_group = None
            if enable_hierarchical_comm:
                # TODO: move to plugin
                self.ep_intra_src_rank, *self.ep_hierarchical_group = create_ep_hierarchical_group(
                    get_ep_group_ranks(self.experts)
                )
            self.dp_group = get_dp_group(self.experts)
        else:
            self.ep_group = None
            self.dp_group = None
        self.num_local_experts = self.experts.num_local_experts

        # load balance
        self.enable_load_balance = enable_load_balance
        if self.enable_load_balance == True:
            self.load_balancer = LoadBalancer(
                experts=self.experts,
                gate=self.gate_weight,
                local_expert_num=self.num_local_experts,
                expert_num=self.num_experts,
                ep_group=self.ep_group,
                dp_group=self.dp_group,
                tolerance=load_balance_tolerance,
                beam_width=load_balance_beam_width,
                group_swap_factor=load_balance_group_swap_factor,
            )

        # init param
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.normal_(self.gate_weight, std=math.sqrt(0.1 / self.hidden_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): The input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # reshape the input tokens
        tokens = inputs.reshape(-1, self.hidden_size)

        # the data type of the inputs in the gating should be fp32
        gate_logits = F.linear(tokens, self.gate_weight)
        gate_output = gate_logits.to(torch.float)

        # update expert load
        if self.enable_load_balance == True:
            with torch.no_grad():
                # TODO: optimize computation
                expert_load = torch.topk(gate_output, k=self.topk, dim=-1)[1]
                # TODO: bincount introduces synchronize, fix it
                expert_load = torch.bincount(expert_load.view(-1))
                self.load_balancer.update_load(expert_load)

        # the result from the router
        used_capacity, *route_result_list = self.router(
            inputs=gate_output,
            use_kernel=self.enable_kernel,
            ep_group=self.ep_group,
            use_loss=self.router_loss,
            use_norm=self.router_norm,
        )

        # dispatch_data: (num_experts, capacity, hidden_size)
        if self.enable_kernel:
            dispatch_data = MoeDispatch.apply(tokens, *route_result_list[1:])
            dispatch_data = dispatch_data.reshape(self.num_experts, -1, self.hidden_size)
        else:
            sec_mask_f = route_result_list[1].type_as(inputs)
            dispatch_data = torch.matmul(sec_mask_f.permute(1, 2, 0), tokens)

        # expert_output: (num_groups, num_experts, capacity, hidden_size)
        if self.parallel == "EP":
            expert_output = self._ep_process(dispatch_data, used_capacity, overlap=self.enable_comm_overlap)
        elif self.parallel == "TP":
            expert_output = self._tp_process(dispatch_data, used_capacity, overlap=self.enable_comm_overlap)
        elif self.parallel is None:
            expert_output = self._local_process(dispatch_data)
        else:
            raise NotImplementedError(
                "This kind of communication has not been implemented yet.\n" "Please use Experts build function."
            )

        if self.enable_kernel:
            expert_output = expert_output.reshape(-1, self.hidden_size)
            ans = MoeCombine.apply(expert_output, *route_result_list)
        else:
            combine_weights = route_result_list[0].type_as(inputs)
            combine_weights = combine_weights.view(combine_weights.shape[0], -1)
            expert_output = expert_output.view(-1, expert_output.shape[-1])
            ans = torch.matmul(combine_weights, expert_output)

        ans = ans.reshape(inputs.shape)

        if self.return_gate_logits:
            return ans, gate_logits
        else:
            return ans

    def _local_process(self, expert_in: torch.Tensor) -> torch.Tensor:
        expert_in = expert_in.unsqueeze(0)
        expert_out = self.experts(expert_in)
        return expert_out

    def _ep_process(
        self, dispatch_data: torch.Tensor, used_capacity: torch.Tensor, overlap: bool = False
    ) -> torch.Tensor:
        """
        Expert Parallel

        Args:
            dispatch_data (torch.Tensor): (num_experts, capacity, hidden_size)

        Returns:
            torch.Tensor: (num_experts, capacity, hidden_size)
        """
        if not overlap or dist.get_world_size(self.ep_group) == 1:
            if self.ep_hierarchical_group is not None:
                expert_input = HierarchicalAllToAll.apply(
                    dispatch_data, self.ep_hierarchical_group, self.ep_intra_src_rank
                )
                expert_input = expert_input.reshape(self.ep_size, self.num_local_experts, -1, self.hidden_size)
                expert_output = self.experts(expert_input)
                expert_output = HierarchicalAllToAll.apply(
                    expert_output, self.ep_hierarchical_group, self.ep_intra_src_rank
                )
                return expert_output
            else:
                expert_input = AllToAll.apply(dispatch_data, self.ep_group, False)[0]
                expert_input = expert_input.reshape(self.ep_size, self.num_local_experts, -1, self.hidden_size)
                expert_output = self.experts(expert_input)
                expert_output = AllToAll.apply(expert_output, self.ep_group, False)[0]
                return expert_output
        else:

            @dataclasses.dataclass
            class Capsule:
                data: torch.Tensor
                handle: Any = None

            NUM_CHUNK = 4
            NUM_STAGES = 4

            assert dispatch_data.shape[1] % NUM_CHUNK == 0, "arbitrary chunk num is not supported yet"
            chunk_size = dispatch_data.shape[1] // NUM_CHUNK
            input_shape = (self.ep_size, self.num_local_experts, -1, self.hidden_size)
            dispatch_data = dispatch_data.reshape(*input_shape)
            chunk_data = torch.split(dispatch_data, chunk_size, dim=2)
            output = torch.empty_like(dispatch_data)

            offset = 0
            _expert_in, expert_in, _expert_out, expert_out = None, None, None, None

            for i in range(NUM_CHUNK + NUM_STAGES - 1):
                if expert_out is not None:
                    expert_out.handle.wait()
                    output[:, :, offset : offset + chunk_size, :] = expert_out.data
                    offset += chunk_size
                    expert_out = None

                # all2all last output
                if _expert_out is not None:
                    expert_out = Capsule(
                        *AllToAll.apply(_expert_out.data, self.ep_group, True),
                    )
                    _expert_out = None

                # all2all next input
                if 0 <= i < NUM_CHUNK:
                    _expert_in = Capsule(*AllToAll.apply(chunk_data[i].contiguous(), self.ep_group, True))

                # compute
                if expert_in is not None:
                    expert_in.handle.wait()
                    _expert_out = Capsule(data=self.experts(expert_in.data), handle=None)
                    expert_in = None

                if _expert_in is not None:
                    expert_in = _expert_in
                    _expert_in = None

            return output

    def _tp_process(
        self, dispatch_data: torch.Tensor, used_capacity: torch.Tensor, overlap: bool = False
    ) -> torch.Tensor:
        """
        without overlap:
                   |    C    |
        |     A    |         |    R    |

        with overlap:
              |    C1   ||    C2   ||    C3   ||    C4   |
        | A1 || A2 |     | R1 | A3 || R2 | A4 || R3 |     | R4 |

        where C is computation, A is all gather, R is reduce scatter.

        Args:
            dispatch_data (torch.Tensor): (num_experts, capacity, hidden_size)

        Returns:
            torch.Tensor: (num_experts, capacity, hidden_size)
        """
        if not overlap or dist.get_world_size(self.ep_group) == 1:
            expert_in = AllGather.apply(dispatch_data, self.ep_group, False)[0]
            expert_out = self.experts(expert_in)
            expert_out = ReduceScatter.apply(expert_out, self.ep_group, False)[0]
            return expert_out
        else:

            @dataclasses.dataclass
            class Capsule:
                data: torch.Tensor
                handle: Any
                indices: Tuple

            NUM_CHUNK = 4
            NUM_STAGES = 4

            assert (
                dispatch_data.shape[0] % NUM_CHUNK == 0
            ), "arbitrary chunk num is not supported yet, please use chunk num that can divide num_experts"
            chunk_size = dispatch_data.shape[0] // NUM_CHUNK
            chunk_data = torch.split(dispatch_data, chunk_size, dim=0)
            output = torch.empty_like(dispatch_data)

            def get_chunk_slice(idx: int, chunk_size: int) -> Tuple[slice]:
                return (slice(idx * chunk_size, (idx + 1) * chunk_size),)

            _expert_in, expert_in, _expert_out, expert_out = None, None, None, None

            for i in range(NUM_CHUNK + NUM_STAGES - 1):
                if expert_out is not None:
                    expert_out.handle.wait()
                    output[expert_out.indices] = expert_out.data
                    expert_out = None

                # reduce scatter last output
                if _expert_out is not None:
                    expert_out = Capsule(
                        *ReduceScatter.apply(_expert_out.data, self.ep_group, True),
                        indices=_expert_out.indices,
                    )
                    _expert_out = None

                # all gather next input
                if 0 <= i < NUM_CHUNK:
                    _expert_in = Capsule(
                        *AllGather.apply(chunk_data[i].contiguous(), self.ep_group, True),
                        indices=get_chunk_slice(i, chunk_size),
                    )

                # compute
                if expert_in is not None:
                    expert_in.handle.wait()
                    _expert_out = Capsule(
                        self.experts(expert_in.data, expert_in.indices),
                        handle=None,
                        indices=expert_in.indices,
                    )
                    expert_in = None

                if _expert_in is not None:
                    expert_in = _expert_in
                    _expert_in = None

            return output


def apply_load_balance(model: nn.Module, optim: Any) -> None:
    """
    apply load balance to every experts in the model
    """

    def _apply_recursive(module: nn.Module):
        for _, sub_module in module.named_children():
            if isinstance(sub_module, SparseMLP):
                if sub_module.enable_load_balance == True:
                    sub_module.load_balancer.balance_load(optim)
            _apply_recursive(sub_module)

    torch.cuda.empty_cache()
    _apply_recursive(model)
    torch.cuda.empty_cache()
