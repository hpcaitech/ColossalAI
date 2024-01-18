import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close
from transformers.activations import ACT2FN

tokens, n_experts = 7, 4
hidden_size = 8
top_k = 2


class MixtralBLockSparseTop2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act="silu"):
        super().__init__()
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states, routing_weights):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return routing_weights * current_hidden_states


class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_local_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.hidden_dim = hidden_size
        self.ffn_dim = intermediate_size
        self.num_experts = num_local_experts
        self.top_k = num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [MixtralBLockSparseTop2MLP(hidden_size, intermediate_size) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class EPMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ep_size = dist.get_world_size()
        ep_rank = dist.get_rank()
        assert self.num_experts % ep_size == 0
        num_experts_per_ep = self.num_experts // ep_size
        expert_start_idx = ep_rank * num_experts_per_ep

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
        dispatch_states = hidden_states.repeat(top_k, 1)[selected_experts_idx]
        input_split_sizes = selected_experts.bincount()
        output_split_sizes = torch.zeros_like(input_split_sizes)
        dist.all_to_all_single(output_split_sizes, input_split_sizes)
        output_states = torch.zeros(output_split_sizes.sum().item(), hidden_size).cuda()

        input_split_list = input_split_sizes.view(ep_size, num_experts_per_ep).sum(dim=-1).tolist()
        output_split_list = output_split_sizes.view(ep_size, num_experts_per_ep).sum(dim=-1).tolist()
        dist.all_to_all_single(output_states, dispatch_states, output_split_list, input_split_list)
        # compute expert output
        if num_experts_per_ep == 1:
            # no need to split
            expert = self.experts[expert_start_idx]
            output_states = expert.act_fn(expert.w1(output_states)) * expert.w3(output_states)
            output_states = expert.w2(output_states)
        else:
            output_states_splits = output_states.split(output_split_sizes.tolist())
            output_states_list = []
            for i, split_states in enumerate(output_states_splits):
                expert = self.experts[expert_start_idx + i % num_experts_per_ep]
                split_states = expert.act_fn(expert.w1(split_states)) * expert.w3(split_states)
                split_states = expert.w2(split_states)
                output_states_list.append(split_states)
            output_states = torch.cat(output_states_list)
        dist.all_to_all_single(dispatch_states, output_states, input_split_list, output_split_list)

        recover_experts_idx = torch.empty_like(selected_experts_idx)
        recover_experts_idx[selected_experts_idx] = torch.arange(
            selected_experts_idx.size(0), device=selected_experts_idx.device
        )
        dispatch_states = dispatch_states[recover_experts_idx]
        k_hidden_states = dispatch_states.chunk(top_k)
        output_states = k_hidden_states[0] * routing_weights[:, 0, None]
        for i in range(1, top_k):
            output_states += k_hidden_states[i] * routing_weights[:, i, None]
        output_states = output_states.reshape(batch_size, sequence_length, hidden_dim)
        return output_states, router_logits


if __name__ == "__main__":
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(dist.get_rank())
    torch.manual_seed(0)
    model = MixtralSparseMoeBlock(hidden_size, hidden_size * 2, n_experts, top_k).cuda()
    x = torch.rand(1, tokens, hidden_size).cuda()
    orig_output, orig_logits = model(x)
    model.__class__ = EPMixtralSparseMoeBlock
    ep_output, ep_logits = model(x)
    assert_close(orig_logits, ep_logits)
    assert_close(orig_output, ep_output)
