import torch
import torch.distributed as dist
import torch.nn.functional as F

# from colossalai.tensor.moe_tensor.moe_info import MoeParallelInfo
from torch.distributed import ProcessGroup
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from colossalai.lazy import LazyInitContext
from colossalai.moe._operation import MoeInGradScaler, MoeOutGradScaler, all_to_all_uneven
from colossalai.shardformer.shard.utils import set_tensors_to_none


class EPMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    def __init__(self, config):
        self.moe_info = None
        super().__init__(config)

    def setup_ep(self, ep_group: ProcessGroup):
        ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
        self.ep_rank = dist.get_rank(ep_group) if ep_group is not None else 0
        assert self.num_experts % self.ep_size == 0
        self.ep_group = ep_group
        self.num_experts_per_ep = self.num_experts // self.ep_size
        self.expert_start_idx = self.ep_rank * self.num_experts_per_ep
        held_experts = self.experts[self.expert_start_idx : self.expert_start_idx + self.num_experts_per_ep]
        set_tensors_to_none(self.experts, exclude=set(held_experts))
        for p in self.experts.parameters():
            # set_moe_tensor_info(p, moe_info)
            p.ep_group = ep_group

    @staticmethod
    def from_native_module(module: MixtralSparseMoeBlock, *args, **kwargs) -> "EPMixtralSparseMoeBlock":
        LazyInitContext.materialize(module)
        module.__class__ = EPMixtralSparseMoeBlock
        # if "ep_group" in kwargs:
        assert "ep_group" in kwargs, "You should pass ep_group in SubModuleReplacementDescription via shard_config!!"
        module.setup_ep(kwargs["ep_group"])
        return module

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        selected_experts = selected_experts.t().reshape(-1)
        selected_experts_idx = selected_experts.argsort()
        dispatch_states = hidden_states.repeat(self.top_k, 1)[selected_experts_idx]
        input_split_sizes = selected_experts.bincount(minlength=self.num_experts)
        output_split_sizes = torch.zeros_like(input_split_sizes)
        dist.all_to_all_single(output_split_sizes, input_split_sizes, group=self.ep_group)

        input_split_list = input_split_sizes.view(self.ep_size, self.num_experts_per_ep).sum(dim=-1).tolist()
        output_split_list = output_split_sizes.view(self.ep_size, self.num_experts_per_ep).sum(dim=-1).tolist()
        output_states, _ = all_to_all_uneven(dispatch_states, input_split_list, output_split_list, self.ep_group)
        # compute expert output
        output_states = MoeInGradScaler.apply(output_states, self.ep_size)
        if output_states.size(0) > 0:
            if self.num_experts_per_ep == 1:
                # no need to split
                expert = self.experts[self.expert_start_idx]
                output_states = expert.act_fn(expert.w1(output_states)) * expert.w3(output_states)
                output_states = expert.w2(output_states)
            else:
                output_states_splits = output_states.split(output_split_sizes.tolist())
                output_states_list = []
                for i, split_states in enumerate(output_states_splits):
                    if split_states.size(0) == 0:
                        continue
                    expert = self.experts[self.expert_start_idx + i % self.num_experts_per_ep]
                    split_states = expert.act_fn(expert.w1(split_states)) * expert.w3(split_states)
                    split_states = expert.w2(split_states)
                    output_states_list.append(split_states)
                output_states = torch.cat(output_states_list)
        output_states = MoeOutGradScaler.apply(output_states, self.ep_size)
        dispatch_states, _ = all_to_all_uneven(output_states, output_split_list, input_split_list, self.ep_group)
        recover_experts_idx = torch.empty_like(selected_experts_idx)
        recover_experts_idx[selected_experts_idx] = torch.arange(
            selected_experts_idx.size(0), device=selected_experts_idx.device
        )
        dispatch_states = dispatch_states[recover_experts_idx]
        k_hidden_states = dispatch_states.chunk(self.top_k)
        output_states = k_hidden_states[0] * routing_weights[:, 0, None]
        for i in range(1, self.top_k):
            output_states += k_hidden_states[i] * routing_weights[:, i, None]
        output_states = output_states.reshape(batch_size, sequence_length, hidden_dim)
        return output_states, router_logits
