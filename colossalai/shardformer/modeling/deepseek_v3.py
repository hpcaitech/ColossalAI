import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.lazy import LazyInitContext
from colossalai.moe._operation import (
    DPGradScalerIn,
    DPGradScalerOut,
    EPGradScalerIn,
    EPGradScalerOut,
    all_to_all_uneven,
)
from colossalai.shardformer.layer.linear import ParallelModule
from colossalai.shardformer.shard.utils import set_tensors_to_none
from colossalai.tensor.moe_tensor.api import set_moe_tensor_ep_group


class EpDeepseekV3MoE(ParallelModule):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        raise RuntimeError(f"Please use `from_native_module` to create an instance of {self.__class__.__name__}")

    def setup_process_groups(
        self,
        moe_dp_group: ProcessGroup,
        ep_group: ProcessGroup,
    ):
        assert moe_dp_group is not None
        assert ep_group is not None

        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        self.num_experts = self.config.n_routed_experts
        assert self.num_experts % self.ep_size == 0

        self.ep_group = ep_group
        self.num_experts_per_ep = self.num_experts // self.ep_size
        self.experts_per_rank = self.num_experts_per_ep
        self.expert_start_idx = self.ep_rank * self.num_experts_per_ep
        held_experts = self.experts[self.expert_start_idx : self.expert_start_idx + self.num_experts_per_ep]

        set_tensors_to_none(self.experts, exclude=set(held_experts))

        # setup moe_dp group
        self.moe_dp_group = moe_dp_group
        self.moe_dp_size = dist.get_world_size(moe_dp_group)

        for p in self.experts.parameters():
            set_moe_tensor_ep_group(p, ep_group)

    @staticmethod
    def from_native_module(
        module,
        moe_dp_group: ProcessGroup,
        ep_group: ProcessGroup,
        *args,
        **kwargs,
    ) -> "EpDeepseekV3MoE":
        LazyInitContext.materialize(module)
        if module.__class__.__name__ == "DeepseekV3MLP":
            return module
        module.__class__ = EpDeepseekV3MoE
        module.setup_process_groups(moe_dp_group, ep_group)
        return module

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_forward(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    def moe_forward(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=self.ep_group)

            output_splits = tokens_per_expert_group.view(self.ep_size, -1).sum(1).tolist()
            input_split_sizes = tokens_per_ep_rank.tolist()

            gathered_tokens, _ = all_to_all_uneven(sorted_tokens, input_split_sizes, output_splits, self.ep_group)
            tokens_per_expert_post_gather = tokens_per_expert_group.view(self.ep_size, self.experts_per_rank).sum(dim=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather

            # moe-dp related code
            activate_experts = tokens_per_expert_post_gather > 0
            activate_experts = activate_experts.int()
            dist.all_reduce(activate_experts, group=self.moe_dp_group)

            # ep related code
            sorted_tokens = EPGradScalerIn.apply(sorted_tokens, self.ep_size)

        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            # moe-dp related code
            tokens_for_this_expert = DPGradScalerIn.apply(tokens_for_this_expert, self.moe_dp_size, activate_experts[i])
            expert_out = expert(tokens_for_this_expert)
            # moe-dp related code
            expert_out = DPGradScalerOut.apply(expert_out, self.moe_dp_size, activate_experts[i])
            outputs.append(expert_out)
            start_idx = end_idx

        if len(outputs) > 0:
            outs = torch.cat(outputs, dim=0)
        else:
            assert sorted_tokens.numel() == 0, f"sorted_tokens: should be empty, but got {sorted_tokens.shape}"
            outs = sorted_tokens

        if self.ep_size > 1:
            outs = EPGradScalerOut.apply(outs, self.ep_size)
            new_x = torch.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens, _ = all_to_all_uneven(new_x, output_splits, input_split_sizes, self.ep_group)
            outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            (new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype) * topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )

        return final_out
