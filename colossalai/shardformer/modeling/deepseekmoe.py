import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.lazy import LazyInitContext
from colossalai.moe._operation import MoeInGradScaler, MoeOutGradScaler, all_to_all_uneven
from colossalai.shardformer.shard.utils import set_tensors_to_none

from .deepseek_moe_16b_base.configuration_deepseek import DeepseekConfig

# from colossalai.tensor.moe_tensor.moe_info import MoeParallelInfo
from .deepseek_moe_16b_base.modeling_deepseek import AddAuxiliaryLoss, DeepseekMoE


class EPDeepseekMoE(DeepseekMoE):
    def __init__(self, config: DeepseekConfig):
        super().__init__(config)

    def setup_ep(self, ep_group: ProcessGroup):
        ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
        self.ep_rank = dist.get_rank(ep_group) if ep_group is not None else 0
        self.num_experts = self.config.n_routed_experts
        assert self.num_experts % self.ep_size == 0
        self.ep_group = ep_group
        self.num_experts_per_ep = self.num_experts // self.ep_size
        self.expert_start_idx = self.ep_rank * self.num_experts_per_ep
        held_experts = self.experts[self.expert_start_idx : self.expert_start_idx + self.num_experts_per_ep]
        set_tensors_to_none(self.experts, exclude=set(held_experts))
        for p in self.experts.parameters():
            p.ep_group = ep_group

    @staticmethod
    def from_native_module(module: DeepseekMoE, *args, **kwargs) -> "EPDeepseekMoE":
        LazyInitContext.materialize(module)
        module.__class__ = EPDeepseekMoE
        assert "ep_group" in kwargs, "You should pass ep_group in SubModuleReplacementDescription via shard_config!!"
        module.setup_ep(kwargs["ep_group"])
        return module

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape

        topk_experts_idx, topk_experts_weight, aux_loss = self.gate(hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])  # [t0, t1, t2 ...]
        hidden_states = hidden_states.repeat_interleave(
            self.num_experts_per_tok, dim=0
        )  # after repeat_interleave: [t0 t0 t1 t1 t2 t2 ... ]

        flat_topk_experts_idx = topk_experts_idx.view(-1)  # [e0 e1 e2 ...]
        # The elements of flat_topk_token_idx are token ids, which are arranged in ascending order of expert ids.
        flat_topk_token_idx = flat_topk_experts_idx.argsort()

        # Now we adjust the order of the hidden states, also in ascending order of expert id
        dispatch_states = hidden_states[flat_topk_token_idx]
        input_split_sizes = flat_topk_experts_idx.bincount(minlength=self.num_experts)  # [n0, n1, n2, n3]
        print(f"{input_split_sizes=}")
        output_split_sizes = torch.zeros_like(input_split_sizes)

        # [n0, n1, n2, n3] [m0, m1, m2, m3] -> [n0, n1, m0, m1] [n2, n3, m2, m3]
        dist.all_to_all_single(output_split_sizes, input_split_sizes, group=self.ep_group)

        input_split_list = input_split_sizes.view(self.ep_size, self.num_experts_per_ep).sum(dim=-1).tolist()
        output_split_list = output_split_sizes.view(self.ep_size, self.num_experts_per_ep).sum(dim=-1).tolist()
        output_states, _ = all_to_all_uneven(dispatch_states, input_split_list, output_split_list, self.ep_group)
        output_states = MoeInGradScaler.apply(output_states, self.ep_size)

        if output_states.size(0) > 0:
            if self.num_experts_per_ep == 1:
                expert = self.experts[self.expert_start_idx]
                output_states = expert(output_states)
            else:
                output_states_splits = output_states.split(output_split_sizes.tolist())
                output_states_list = []
                for i, split_states in enumerate(output_states_splits):
                    if split_states.size(0) == 0:  # no token routed to this experts
                        continue
                    expert = self.experts[self.expert_start_idx + i % self.num_experts_per_ep]
                    split_states = expert(split_states)
                    output_states_list.append(split_states)
                output_states = torch.cat(output_states_list)  # (4, h) (8, h)
        output_states = MoeOutGradScaler.apply(output_states, self.ep_size)
        dispatch_states, _ = all_to_all_uneven(
            output_states, output_split_list, input_split_list, self.ep_group
        )  # 专家处理完对应token的输出，要返还回去给别的rank
        recover_token_idx = torch.empty_like(flat_topk_token_idx)  # (6,)
        recover_token_idx[flat_topk_token_idx] = torch.arange(
            flat_topk_token_idx.size(0), device=flat_topk_token_idx.device
        )

        output_hidden_states = dispatch_states[recover_token_idx]  # t0 t0 t1 t1 t2 t2
        output_hidden_states = output_hidden_states.view(-1, self.num_experts_per_tok, orig_shape[-1])
        output_hidden_states = (output_hidden_states * topk_experts_weight[:, :, None]).sum(dim=-2)  # (BS, h)
        output_hidden_states = output_hidden_states.view(*orig_shape)
        output_hidden_states = AddAuxiliaryLoss.apply(output_hidden_states, aux_loss)
        if self.config.n_shared_experts is not None:
            output_hidden_states = output_hidden_states + self.shared_experts(identity)
        return output_hidden_states
